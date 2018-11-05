#!/usr/bin/env python
import argparse
import copy
import functools
import json
import math
import operator
import os
import torch
import torchvision

from torch import nn
from plnn.modules import View
from plnn.model import simplify_network, dump_rlv

MNIST_FOLDER = './'

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

def train_mnist(train_data, train_labels,
                test_data, test_labels,
                net, use_cuda, only_test=False):

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=512, shuffle=True
    )

    if use_cuda:
        net.cuda()
    lr = 0.1
    momentum = 0.5
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    if not only_test:
        # Training
        net.train()
        total_loss = 0
        best_total_loss = float('inf')
        epoch_idx = 0
        while lr > 1e-8:
            epoch_idx = 0
            while True:
                epoch_idx += 1
                total_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output = net(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.data.item()

                    if batch_idx % 10 == 0:
                        # print(f"Epoch {epoch_idx} -- {100 * batch_idx / len(train_loader)}% --Loss: {loss.data[0]}")
                        pass
                if total_loss >= best_total_loss:
                    break
                else:
                    best_total_loss = total_loss
            lr /= 5

    # Testing
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=64, shuffle=True
    )
    net.eval()
    accuracies = {}
    loaders = [('Train', train_loader), ('Test', test_loader)]
    for loader_name, loader in loaders:
        correct = 0
        for data, target in loader:
            if use_cuda:
                data = data.cuda()
            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.cpu().eq(target.view_as(pred)).sum()
        accuracy = 100 * correct / len(loader.dataset)
        accuracies[loader_name] = accuracy.item()
        # print(f"{loader_name} set Accuracy: {accuracy}")
    return accuracies

def load_mnist():
    # Setup up folder structure if this hasn't been done.
    mnist_data = os.path.join(MNIST_FOLDER, 'data')
    os.makedirs(mnist_data, exist_ok=True)
    mnist_train = torchvision.datasets.MNIST(mnist_data,
                                             train=True,
                                             download=True)
    mnist_test = torchvision.datasets.MNIST(mnist_data,
                                            train=False,
                                            download=False)
    mnist_train_tensor = mnist_train.train_data.float().unsqueeze(1) / 255
    mnist_test_tensor = mnist_test.test_data.float().unsqueeze(1) / 255
    mnist_train_labels = mnist_train.train_labels
    mnist_test_labels = mnist_test.test_labels

    return mnist_train_tensor, mnist_train_labels, \
        mnist_test_tensor, mnist_test_labels




def load_PCA_mnist():
    # Setup up folder structure if this hasn't been done.
    mnist_data = os.path.join(MNIST_FOLDER, 'data')
    os.makedirs(mnist_data, exist_ok=True)
    mnist_pca_tensors = os.path.join(mnist_data, 'PCAMnist.pth')

    # If we don't already have PCAedMnist, build it
    if not os.path.exists(mnist_pca_tensors):
        mnist_train = torchvision.datasets.MNIST(mnist_data,
                                                 train=True,
                                                 download=True)
        mnist_test = torchvision.datasets.MNIST(mnist_data,
                                                train=False,
                                                download=False)
        mnist_train_tensor = mnist_train.train_data.double().view(len(mnist_train), 28*28) / 255
        mnist_test_tensor = mnist_test.test_data.double().view(len(mnist_test), 28*28) / 255
        mnist_train_labels = mnist_train.train_labels
        mnist_test_labels = mnist_test.test_labels

        mnist_train_mean = torch.mean(mnist_train_tensor, dim=0)
        nomean_mnist_train = mnist_train_tensor - mnist_train_mean.unsqueeze(0)
        nomean_mnist_test = mnist_test_tensor - mnist_train_mean.unsqueeze(0)

        U, S, V = torch.svd(nomean_mnist_train)

        # Get the decomposition in the basis of the eigenvectors
        basis = torch.mm(torch.diag(S+1e-12), torch.t(V))
        trainInBasis = U
        testInBasis = torch.mm(torch.mm(nomean_mnist_test, V), torch.diag(1/(S+1e-12)))

        # Ensure that we have the correct decomposition /
        # we can recover original samples
        rebuilt_nomean_train = torch.mm(trainInBasis, basis)
        rebuilt_nomean_test = torch.mm(testInBasis, basis)
        assert (rebuilt_nomean_train - nomean_mnist_train).abs().max() < 1e-7
        assert (rebuilt_nomean_test - nomean_mnist_test).abs().max() < 1e-7
        rebuilt_mnist_train = rebuilt_nomean_train + mnist_train_mean.unsqueeze(0)
        rebuilt_mnist_test = rebuilt_nomean_test + mnist_train_mean.unsqueeze(0)
        assert (rebuilt_mnist_train - mnist_train_tensor).abs().max() < 1e-7
        assert (rebuilt_mnist_test - mnist_test_tensor).abs().max() < 1e-7

        PCA_params = {
            "mean": mnist_train_mean,
            "basis": basis,
            "trainInBasis": trainInBasis,
            "trainLabels": mnist_train_labels,
            "testInBasis": testInBasis,
            "testLabels": mnist_test_labels
        }
        torch.save(PCA_params, mnist_pca_tensors)
    else:
        PCA_params = torch.load(mnist_pca_tensors)

        trainInBasis = PCA_params["trainInBasis"]
        testInBasis = PCA_params["testInBasis"]
        mnist_train_labels = PCA_params["trainLabels"]
        mnist_test_labels = PCA_params["testLabels"]
        basis = PCA_params["basis"]
        mnist_train_mean = PCA_params["mean"]

    return trainInBasis, mnist_train_labels, \
        testInBasis, mnist_test_labels, \
        basis, mnist_train_mean

def flatten_layers(layer_list):
    input_shape = (layer_list[0].in_features,)
    new_layers = []
    for layer in layer_list:
        if type(layer) is nn.Linear:
            assert len(input_shape) == 1
            assert input_shape[0] == layer.in_features
            new_layers.append(layer)
            input_shape = (layer.out_features,)
        elif type(layer) is View:
            nb_element = prod(input_shape)
            view_element = prod(layer.out_shape)
            assert view_element == nb_element
            input_shape = layer.out_shape
        elif type(layer) is nn.Conv2d:
            in_chan, in_height, in_width = input_shape
            out_chan = layer.out_channels
            assert layer.padding == (0, 0)
            assert layer.dilation == (1, 1)
            assert layer.stride[0] == layer.stride[1]
            assert layer.in_channels == in_chan
            out_height = math.floor((in_height - layer.kernel_size[0])// layer.stride[0] + 1)
            out_width = math.floor((in_width - layer.kernel_size[1])// layer.stride[1] + 1)

            flat_nb_in = in_chan * in_height * in_width
            flat_nb_out = out_chan * out_height * out_width
            eq_layer = nn.Linear(flat_nb_in, flat_nb_out)
            conv_weight = layer.weight.data
            conv_bias = layer.bias.data
            eqlin_weight = eq_layer.weight.data
            eqlin_weight.fill_(0)
            eqlin_bias = eq_layer.bias.data
            eqlin_bias.fill_(0)
            # Brain too small for smart stuff, just for-loop it
            for out_c in range(out_chan):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        out_flat_idx = out_w + out_h * out_width + out_c * out_height * out_width

                        # Figure out which are the input idx that needs to be considered
                        for in_c in range(in_chan):
                            for k_h in range(layer.kernel_size[0]):
                                for k_w in range(layer.kernel_size[1]):
                                    in_h = layer.stride[0]*out_h + k_h
                                    in_w = layer.stride[1]*out_w + k_w
                                    in_flat_idx = in_w + in_h * in_width + in_c * in_height * in_width

                                    eqlin_weight[out_flat_idx, in_flat_idx] = conv_weight[out_c, in_c, k_h, k_w]
                        eqlin_bias[out_flat_idx] = conv_bias[out_c]
            input_shape = (out_chan, out_height, out_width)
            new_layers.append(eq_layer)
        elif type(layer) is nn.ReLU:
            new_layers.append(layer)
        elif type(layer) is nn.MaxPool2d:
            # Only doing this special case because I'm lazy and that's all I need for now
            # This has the advantage that each element of the input appears once in the output
            assert layer.kernel_size == layer.stride
            # We are going to make it work by combining a linear layer (to shuffle things around)
            # and a MaxPool1d to actually do the maxpooling
            nb_elt = prod(input_shape)
            nb_chan, height, width = input_shape
            shuffle_linear = nn.Linear(nb_elt, nb_elt)
            shuffle_linear_weight = shuffle_linear.weight.data
            shuffle_linear_bias = shuffle_linear.bias.data
            shuffle_linear_weight.fill_(0)
            shuffle_linear_bias.fill_(0)
            nb_pool_h = height // layer.kernel_size
            nb_pool_w = width // layer.kernel_size
            nb_in_pool = layer.kernel_size * layer.kernel_size
            for out_c in range(nb_chan):
                for group_h in range(nb_pool_h):
                    for group_w in range(nb_pool_w):

                        for in_pool_h in range(layer.kernel_size):
                            for in_pool_w in range(layer.kernel_size):
                                out_idx = (((out_c * nb_pool_h + group_h) * nb_pool_w + group_w) * layer.kernel_size + in_pool_h) * layer.kernel_size + in_pool_w

                                in_c = out_c
                                in_h = group_h * layer.kernel_size + in_pool_h
                                in_w = group_w * layer.kernel_size + in_pool_w

                                in_idx = (in_c * height + in_h) * width + in_w
                                shuffle_linear_weight[out_idx, in_idx] = 1
            new_layers.append(shuffle_linear)
            # Now that we have re-arranged everything, we can do a Maxpooling
            new_layers.append(View((1, nb_elt)))
            new_layers.append(nn.MaxPool1d(nb_in_pool))
            input_shape = (nb_chan, nb_pool_h, nb_pool_w)
            nb_out_elt = prod(input_shape)
            new_layers.append(View((nb_out_elt,)))
        else:
            raise NotImplementedError
    return new_layers

def main():
    parser = argparse.ArgumentParser(description="Train a network on PCAedMnist"
                                     "eventually building the dataset if necessary")
    parser.add_argument('nb_inputs', type=int)
    parser.add_argument('nb_hidden', type=int)
    parser.add_argument('depth', type=int)
    parser.add_argument('folder_name', type=str)
    parser.add_argument('--task', type=str,
                        choices=['DigitClass', 'OddOrEven'],
                        default='DigitClass')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help="Disables cuda training")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    width = 28
    print(f"Input: 1 x {width} x {width}")
    kernel_size = 8
    kernel_stride = 4
    width = (width - kernel_size) // kernel_stride + 1 # After strided-conv
    nb_channels = max(args.nb_hidden//(width*width), 3)
    net_layers = [nn.Conv2d(1, nb_channels, kernel_size=kernel_size, stride=kernel_stride),
                  # nn.MaxPool2d(2),
                  nn.ReLU()]
    print(f"Post-conv1: {nb_channels} x {width} x {width}")
    # width = width // 2 # Maxpool
    # print(f"Post-MaxPool: {nb_channels} x {width} x {width}")
    kernel_size = 5
    doing_linear = False
    linear_idx = 0

    final_size = 10 if args.task == "DigitClass" else 2
    for i in range(args.depth-1):
        if doing_linear:
            net_layers.append(nn.Linear(args.nb_hidden, args.nb_hidden))
        else:
            new_width = (width - kernel_size) + 1 # After the Conv
            if new_width > 0:
                width = new_width
                new_nb_channels = max(args.nb_hidden//(width*width), 3)
                net_layers.append(nn.Conv2d(nb_channels, new_nb_channels, kernel_size=kernel_size))
                print(f"Post-Conv{i+2}: {new_nb_channels} x {width} x {width}")
                nb_channels = new_nb_channels
            else:
                # Switching to Linear
                doing_linear = True
                flat_dim = width * width * nb_channels
                net_layers.append(View((flat_dim,)))
                net_layers.append(nn.Linear(flat_dim, args.nb_hidden))
                linear_idx += 1
                print(f"Post-FC{linear_idx}: {args.nb_hidden}")
        net_layers.append(nn.ReLU())
    if not doing_linear:
        flat_dim = width*width*nb_channels
        net_layers.append(View((flat_dim,)))
        net_layers.append(nn.Linear(flat_dim, final_size))
    else:
        net_layers.append(nn.Linear(args.nb_hidden, final_size))
    net = nn.Sequential(*net_layers)

    (pca_train_data, train_labels,
     pca_test_data, test_labels,
     basis, mean) = load_PCA_mnist()
    if args.task == "OddOrEven":
        train_labels = train_labels % 2
        test_labels = test_labels % 2

    # Dimensionality reduction
    pca_train_data = pca_train_data.narrow(1, 0, args.nb_inputs)
    pca_test_data = pca_test_data.narrow(1, 0, args.nb_inputs)
    basis = basis.narrow(0, 0, args.nb_inputs)

    # Re-add the mean
    train_data = torch.mm(pca_train_data, basis) + mean.unsqueeze(0)
    test_data = torch.mm(pca_test_data, basis) + mean.unsqueeze(0)
    # Reshape into images
    train_data = train_data.view(train_data.shape[0], 1 , 28, 28).float()
    test_data = test_data.view(test_data.shape[0], 1 , 28, 28).float()

    mnist_trained = os.path.join(MNIST_FOLDER, 'nets')
    net_folder_path = args.folder_name
    os.makedirs(net_folder_path, exist_ok=True)

    net_path = os.path.join(net_folder_path, "PCA-MNIST.net")
    accuracy_path = os.path.join(net_folder_path, "accuracies.json")
    arch_path = os.path.join(net_folder_path, "arch.txt")
    rlv_path = os.path.join(net_folder_path, "prop.rlv")

    if not os.path.exists(net_path):
        # Train the convnet
        accuracies = train_mnist(train_data, train_labels,
                                 test_data, test_labels,
                                 net, use_cuda)
        print(f"Accuracy after conv training: {accuracies}")
        layers = [layer for layer in net]
        # Include the PCA so that we can run on reduced dimension
        pca_layer = nn.Linear(args.nb_inputs, 28*28)
        pca_layer.weight.data.copy_(torch.t(basis).float())
        pca_layer.bias.data.copy_(mean.float())
        layers.insert(0, pca_layer)
        layers.insert(1, View((1, 28, 28)))
        # Flatten the network
        layers = flatten_layers([layer.cpu() for layer in layers])
        new_net = nn.Sequential(*layers)
        # Verify that we didn't introduce any error
        flat_accuracies = train_mnist(pca_train_data.float(), train_labels,
                                      pca_test_data.float(), test_labels,
                                      new_net, use_cuda, only_test=True)
        assert accuracies['Train'] == flat_accuracies['Train']
        assert accuracies['Test'] == flat_accuracies['Test']

        # Try to train the flattened version and keep the new weights if it improves
        new_layers = [copy.deepcopy(layer) for layer in layers]
        new_net = nn.Sequential(*new_layers)
        new_flat_accuracies = train_mnist(pca_train_data.float(), train_labels,
                                          pca_test_data.float(), test_labels,
                                          new_net, use_cuda)
        if new_flat_accuracies['Train'] >= flat_accuracies['Train']:
            layers = new_layers
            accuracies = new_flat_accuracies
        else:
            accuracies = flat_accuracies
        print(f"Accuracy after flat training: {accuracies}")

        # Simplify the network
        layers = simplify_network([layer.cpu() for layer in layers])
        new_net = nn.Sequential(*layers)
        # Verify that we didn't introduce any obvious error
        simple_accuracies = train_mnist(pca_train_data.float(), train_labels,
                                        pca_test_data.float(), test_labels,
                                        new_net, use_cuda, only_test=True)
        assert simple_accuracies['Train'] == accuracies['Train']
        assert simple_accuracies['Test'] == accuracies['Test']

        # Try to train the simplified version and keep the new weights if it improves
        new_layers = [copy.deepcopy(layer) for layer in layers]
        new_net = nn.Sequential(*new_layers)
        new_simple_accuracies = train_mnist(pca_train_data.float(), train_labels,
                                            pca_test_data.float(), test_labels,
                                            new_net, use_cuda)
        if new_simple_accuracies['Train'] >= accuracies['Train']:
            layers = new_layers
            accuracies = new_simple_accuracies
        print(f"Accuracy after simple training: {accuracies}")

        net = nn.Sequential(*layers)
        torch.save(layers, net_path)
        with open(accuracy_path, 'w') as acc_file:
            json.dump(accuracies, acc_file)
        with open(arch_path, 'w') as arch_file:
            arch_file.write(str(net))
    else:
        layers = torch.load(net_path)
        net = nn.Sequential(*layers)
        # Verify that we didn't introduce any obvious error
        accuracies = train_mnist(pca_train_data.float(), train_labels,
                                 pca_test_data.float(), test_labels,
                                 net, use_cuda, only_test=True)
        with open(accuracy_path, 'r') as acc_file:
            dumped_accuracies = json.load(acc_file)
        assert dumped_accuracies == accuracies

    print("Resulting Network: ")
    print(net)
    print("Accuracies")
    print(accuracies)

    if args.task == "OddOrEven":
        # Add a layer to express:
        # "The difference of score between Even and Odd"
        diff_layer = nn.Linear(2, 1)
        diff_layer.bias.data.fill_(0)
        diff_layer.weight.data[0, 0] = 1
        diff_layer.weight.data[0, 1] = -1
        layers.append(diff_layer)

        # Define the input domain
        feat_max, _ = pca_train_data.max(dim=0)
        feat_min, _ = pca_train_data.min(dim=0)

        domain = torch.stack([feat_min, feat_max], 1)

        with open(rlv_path, 'w') as rlv_outfile:
            dump_rlv(rlv_outfile, layers, domain)


if __name__ == '__main__':
    main()

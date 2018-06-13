#!/usr/bin/env python
import argparse
import os
import time
import torch
import torchvision

from plnn.mip_solver import MIPNetwork
from plnn.modules import View
from plnn.model import load_mat_network
from torch.autograd import Variable
from torch import nn

def main():
    parser = argparse.ArgumentParser(description="Read a .mat file"
                                     "and prove robustness over the dataset.")

    parser.add_argument('mat_infile', type=str,
                        help='.mat file to prove.')
    parser.add_argument('adv_perturb', type=float,
                        help='What proportion to use.')
    parser.add_argument('--sym_bounds', action='store_true')
    parser.add_argument('--use_obj_function', action='store_true')
    parser.add_argument('--interval-analysis', action='store_true')
    args = parser.parse_args()

    layers = load_mat_network(args.mat_infile)

    mnist_data = 'weights/mnistData/'
    mnist_test = torchvision.datasets.MNIST(mnist_data,
                                            train=False,
                                            download=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(mnist_test.test_data, mnist_test.test_labels),
        batch_size=512, shuffle=False
    )

    test_net = nn.Sequential(*layers)
    test_net.eval()

    # correct = 0
    # for data, target in test_loader:
    #     var_data = Variable(data.float().view(-1, 28*28)/255.0, volatile=True)
    #     output = test_net(var_data)
    #     pred = output.data.max(1, keepdim=True)[1]
    #     correct += pred.eq(target.view_as(pred)).sum()
    # accuracy = 100 * correct / len(test_loader.dataset)
    #
    # print(f"Nominal accuracy on MNIST test set: {accuracy} %")

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(mnist_test.test_data, mnist_test.test_labels),
        batch_size=1, shuffle=False
    )

    neg_layer = nn.Linear(1, 1)
    neg_layer.weight.data.fill_(-1)
    neg_layer.bias.data.fill_(0)

    nb_done = 0
    verif_result_folder = 'weights/verif_result'
    for data, target in test_loader:
        example_res_file = verif_result_folder + f"/{nb_done}_sample.txt"
        if os.path.isfile(example_res_file):
            nb_done += 1
            continue


        print(f"{time.ctime()} \tExample {nb_done} starting")
        start = time.time()
        net_layers = layers

        data_lb = data.float().view(28*28)/255.0 - args.adv_perturb
        data_ub = data.float().view(28*28)/255.0 + args.adv_perturb
        domain = torch.clamp(torch.stack((data_lb, data_ub), dim=1),
                             min=0, max=1)

        additional_lin_layer = nn.Linear(10, 9, bias=True)
        lin_weights = additional_lin_layer.weight.data
        lin_weights.fill_(0)
        lin_bias = additional_lin_layer.bias.data
        lin_bias.fill_(0)
        to = 0
        gt = target[0]
        for cls in range(10):
            if cls != gt:
                lin_weights[to, cls] = 1
                lin_weights[to, gt] = -1
                to += 1

        verif_layers = layers + [additional_lin_layer,
                                 View((1, 9)),
                                 nn.MaxPool1d(9),
                                 View((1,)),
                                 neg_layer]


        print(f"{time.ctime()} \tExample {nb_done} has spec.")
        mip_network = MIPNetwork(verif_layers)
        mip_network.setup_model(domain,
                                sym_bounds=args.sym_bounds,
                                use_obj_function=args.use_obj_function,
                                interval_analysis=args.interval_analysis)

        print(f"{time.ctime()} \tExample {nb_done} has MIP setup.")
        sat, solution, nb_visited_states = mip_network.solve(domain)
        end = time.time()


        if sat is False:
            print(f"{time.ctime()} \tExample {nb_done} is Robust.")
            with open(example_res_file, 'w') as res_file:
                res_file.write('Robust\n')
                res_file.write(f'{end-start}\n')
        else:
            print(f"{time.ctime()} \tExample {nb_done} is not Robust.")
            adv_example = Variable(solution[0].view(1, -1))
            pred_on_adv = test_net(adv_example)
            print(f"{time.ctime()} \tPredictions: {pred_on_adv.data}")
            print(f"{time.ctime()} \tGT is: {target}")
            with open(example_res_file, 'w') as res_file:
                res_file.write('NonRobust\n')
                res_file.write(f'{end-start}\n')
                res_file.write(f'Input: {solution[0]}\n')
                res_file.write(f'Pred on adv: {pred_on_adv.data}\n')
                res_file.write(f'GT is : {target}\n')

        print("\n\n")
        nb_done += 1


if __name__ == '__main__':
    main()

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
    parser.add_argument('--modulo', type=int, default=1,
                        help="Use this to specify which part of the samples to run in this process."
                        "This process will only run those for which idx %% modulo == modulo_arg ")
    parser.add_argument('--modulo_arg', type=int, default=0)
    parser.add_argument('--ids_to_run', type=str,
                        help='List of ids to run the verification on')
    parser.add_argument('--dataset', type=str,
                        help='What dataset to run the verification on',
                        default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--result_folder', type=str,
                        help='Where to store the results of the verification')
    args = parser.parse_args()

    layers = load_mat_network(args.mat_infile)

    if args.result_folder is None:
        verif_result_folder = f'weights/{args.dataset}_verif_result'
    else:
        verif_result_folder = args.result_folder

    if args.dataset == 'mnist':
        mnist_data = 'weights/mnistData/'
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(-1))
        ])
        test_dataset = torchvision.datasets.MNIST(mnist_data,
                                                  train=False,
                                                  download=True,
                                                  transform=transform)
    elif args.dataset == 'cifar10':
        cifar_data = 'weights/cifar10Data/'

        center_crop = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((28, 28)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.permute(1,2,0).contiguous().view(-1))
        ])
        test_dataset = torchvision.datasets.CIFAR10(cifar_data,
                                                    train=False,
                                                    download=True,
                                                    transform=center_crop)

    test_loader_256batch = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=256,
                                                       shuffle=False)


    test_net = nn.Sequential(*layers)
    test_net.eval()
    correct = 0
    for sample_idx, (data, target) in enumerate(test_loader_256batch):
        var_data = Variable(data, volatile=True)
        output = test_net(var_data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    accuracy = 100 * correct / len(test_dataset)
    print(f"Nominal accuracy on test set: {accuracy} %")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, shuffle=False
    )
    if args.ids_to_run is not None:
        with open(args.ids_to_run, 'r') as ids_list_file:
            txt = ids_list_file.read()
        str_ids_list = txt.split()
        ids_list = list(map(int, str_ids_list))

    else:
        ids_list = range(len(test_dataset))
    to_process = [idx for cnt, idx in enumerate(ids_list) if cnt % args.modulo == args.modulo_arg]

    # #### DEBUG
    # # with open(verif_result_folder + '/incorrect_ids.txt', 'r') as inc_ids:
    # #     all_inc_ids = []
    # #     for inc_id in inc_ids.readlines():
    # #         all_inc_ids.append(int(inc_id))
    # all_inc_ids = [951]

    # for inc_id in all_inc_ids:
    #     ex_idx = inc_id
    #     print(f"Dealing with example {ex_idx}")
    #     inp = mnist_test.test_data[ex_idx]
    #     label = mnist_test.test_labels[ex_idx]
    #     var_data = Variable(inp.float().view(-1, 28*28)/255.0, volatile=True)
    #     output = test_net(var_data)
    #     _, pred = output.data.max(1, keepdim=True)
    #     print(f"Output is {output}")
    #     print(f"Prediction is {pred[0][0]}")
    #     print(f"Target is {label}")


    #     net_layers = layers
    #     data = mnist_test.test_data[ex_idx]
    #     data_lb = data.float().view(28*28)/255.0 - args.adv_perturb
    #     data_ub = data.float().view(28*28)/255.0 + args.adv_perturb
    #     domain = torch.clamp(torch.stack((data_lb, data_ub), dim=1),
    #                              min=0, max=1)

    #     neg_layer = nn.Linear(1, 1)
    #     neg_layer.weight.data.fill_(-1)
    #     neg_layer.bias.data.fill_(0)

    #     additional_lin_layer = nn.Linear(10, 9, bias=True)
    #     lin_weights = additional_lin_layer.weight.data
    #     lin_weights.fill_(0)
    #     lin_bias = additional_lin_layer.bias.data
    #     lin_bias.fill_(0)
    #     to = 0
    #     gt = label
    #     for cls in range(10):
    #         if cls != gt:
    #             lin_weights[to, cls] = 1
    #             lin_weights[to, gt] = -1
    #             to += 1

    #     verif_layers = layers + [additional_lin_layer,
    #                              View((1, 9)),
    #                              nn.MaxPool1d(9),
    #                              View((1,)),
    #                              neg_layer]

    #     artificial_net = nn.Sequential(*verif_layers)
    #     artificial_net.eval()
    #     art_output = artificial_net(var_data)
    #     print(f"\nTo prove net output is {art_output}")
    #     mip_network = MIPNetwork(verif_layers)
    #     mip_network.setup_model(domain,
    #                             sym_bounds=args.sym_bounds,
    #                             use_obj_function=args.use_obj_function,
    #                             interval_analysis=args.interval_analysis)

    #     sat, solution, nb_visited_states = mip_network.solve(domain)
    #     import IPython; IPython.embed();
    #     import sys; sys.exit()
    #     print(f"Counterexample search for example {ex_idx}: satResult is {sat}\n\n")
    # #### END DEBUG

    neg_layer = nn.Linear(1, 1)
    neg_layer.weight.data.fill_(-1)
    neg_layer.bias.data.fill_(0)
    for sp_idx, (data, target) in enumerate(test_loader):
        if sp_idx not in to_process:
            continue

        example_res_file = verif_result_folder + f"/{sp_idx}_sample.txt"
        if os.path.isfile(example_res_file):
            continue


        print(f"{time.ctime()} \tExample {sp_idx} starting")
        start = time.time()
        net_layers = layers

        data_lb = data - args.adv_perturb
        data_ub = data + args.adv_perturb
        domain = torch.clamp(torch.cat([data_lb, data_ub], dim=0).transpose(1,0),
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


        print(f"{time.ctime()} \tExample {sp_idx} has spec.")
        mip_network = MIPNetwork(verif_layers)
        mip_network.setup_model(domain,
                                sym_bounds=args.sym_bounds,
                                use_obj_function=args.use_obj_function,
                                interval_analysis=args.interval_analysis)

        print(f"{time.ctime()} \tExample {sp_idx} has MIP setup.")
        sat, solution, nb_visited_states = mip_network.solve(domain, timeout=3600)
        end = time.time()


        if sat is False:
            print(f"{time.ctime()} \tExample {sp_idx} is Robust.")
            with open(example_res_file, 'w') as res_file:
                res_file.write('Robust\n')
                res_file.write(f'{end-start}\n')
        elif sat is True:
            print(f"{time.ctime()} \tExample {sp_idx} is not Robust.")
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
        elif sat is None:
            print(f"{time.ctime()} \t Example {sp_idx} failure.")
            with open(example_res_file, 'w') as res_file:
                res_file.write('Verification Failure\n')

        print("\n\n")


if __name__ == '__main__':
    main()

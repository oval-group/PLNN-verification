#!/usr/bin/env python
import argparse
import os
import sh
import sys
import time
import torch
import torchvision

from torch import nn
from torch.autograd import Variable

from plnn.model import dump_nnet, load_mat_network
# torch.set_default_tensor_type('torch.DoubleTensor')

reluplex_bin = '/home/rudy/workspace/PLNN-verification/ReluplexCav2017/check_properties/bin/generic_prover.elf'
reluplex_command = sh.Command(reluplex_bin)

def reluplex_possiblenotrobust(path_to_nnet):
  res_path = path_to_nnet.replace('.nnet', '.out')
  try:
    content = reluplex_command(path_to_nnet, res_path)
  except sh.SignalException_SIGABRT:
    # Reluplex has some cleanup problems sometimes but we don't care as long as
    # it has dumped its results.
    pass
  if os.path.exists(res_path):
    with open(res_path, 'r') as res_file:
      result = res_file.read()
  else:
    print("Reluplex didn't create the result file.")
    print(f"File ran was {path_to_nnet}, should investigate.")
    import IPython; IPython.embed();
    import sys; sys.exit()


  if "UNSAT" in result:
    return False
  elif " SAT" in result:
    return True
  elif " ERROR" in result:
    # Reluplex errored and can't prove that the property is robust
    # It's therefore possible that the property is not robust.
    print("Reluplex errored.")
    return True
  else:
    print("Unknown result for the verification")
    import IPython; IPython.embed();
    import sys; sys.exit()


def main():
  parser = argparse.ArgumentParser(description="Read a .mat file"
                                   "and generate nnet file for Reluplex to prove the robustness of")
  parser.add_argument('mat_infile', type=str,
                      help='.mat file to measure robustness for.')
  parser.add_argument('sample_id', type=int,
                      help='Sample to verify.')
  parser.add_argument('result_folder', type=str,
                      help='Where to store the result')
  parser.add_argument('--dataset', type=str,
                      help='What dataset to run the verification on',
                      default='mnist', choices=['mnist'])
  args = parser.parse_args()

  layers = load_mat_network(args.mat_infile)

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

  working_folder = os.path.join(args.result_folder, str(args.sample_id))
  if not os.path.exists(working_folder):
    os.makedirs(working_folder)

  data, gt = test_dataset[args.sample_id]
  start = time.time()
  var = Variable(data, volatile=True)
  net = nn.Sequential(*layers)
  pred = net(var)
  mx, argmax = pred.max(dim=0)
  pred_label = argmax.data[0]
  if pred_label != gt:
    end = time.time()
    # The prediction is not even correct so the epsilon is just 0
    final_res = os.path.join(working_folder, "binary_search_result.txt")
    print(f"Prediction is: {pred}")
    print(f"While GT is: {gt}")
    with open(final_res, 'w') as res_file:
      res_file.write(f'[0 , 0]\n')
      res_file.write(f'{end - start}\n')
    sys.exit(0)

  known_lb_eps = 0
  known_ub_eps = None
  eps = 0.1
  while True:
    data_lb = data - eps
    data_ub = data + eps
    domain = torch.clamp(torch.stack([data_lb, data_ub], dim=0).transpose(1, 0),
                         min=0, max=1)

    for label in range(10):
      if label != gt:
        additional_layer = nn.Linear(10, 1, bias=True)
        lin_weights = additional_layer.weight.data
        lin_weights.fill_(0)
        lin_bias = additional_layer.bias.data
        lin_bias.fill_(0)
        lin_weights[0, label] = -1
        lin_weights[0, gt] = 1

        # This network encodes the satisfiability of the robustness
        # for the sample
        all_layers = layers + [additional_layer]

        nnet_filename = f"{args.sample_id}sample_{eps}eps_{label}adv.nnet"
        nnet_path = os.path.join(working_folder, nnet_filename)
        with open(nnet_path, 'w') as nnet_outfile:
          dump_nnet(nnet_outfile, all_layers, domain)

        sat = reluplex_possiblenotrobust(nnet_path)
        if sat:
          print(f"{time.ctime()}\tNot robust for adv:{label} and eps:{eps}")
          # There exist a counterexample.
          # Not robust for this epsilon
          # This becomes a valid upper bound on epsilon.
          known_ub_eps = eps
          eps = .5 * (known_lb_eps + known_ub_eps)
          print(f"{time.ctime()} Possible range for epsilon: [{known_lb_eps} , {known_ub_eps}]")
          # No need to check the other labels
          break
        else:
          print(f"{time.ctime()}\tRobust for adv:{label} and eps:{eps}")
    else:
      # All possible label attacks were UNSAT
      # This means that the network was robust for this epsilon.
      # This becomes a valid lower bound on epsilon
      known_lb_eps = eps
      if known_ub_eps is None:
        # We haven't yet found an epsilon for which the model was attackable
        # Just multiply by 2 the radius we consider.
        eps = 2 * eps
      else:
        eps = .5 * (known_lb_eps + known_ub_eps)
      print(f"{time.ctime()} Possible range for epsilon: [{known_lb_eps} , {known_ub_eps}]")
    if (known_ub_eps is not None) and ((known_ub_eps - known_lb_eps) < 1e-5):
      end = time.time()
      final_res = os.path.join(working_folder, "binary_search_result.txt")
      with open(final_res, 'w') as res_file:
        res_file.write(f'[{known_lb_eps} , {known_ub_eps}]\n')
        res_file.write(f'{end - start}\n')
      break

if __name__ == '__main__':
  main()

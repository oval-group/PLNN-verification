import time
import torch

from torch import nn
from torch.autograd import Variable

class HeuristicNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def guess_lower_bound(self, domain, timeout, noprogress_timeout, early_stop=False, use_cuda=False):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        After `timeout` time has elapsed, return the best solution found so far.
        '''
        nb_starts = 0
        best_ub = float('inf')
        best_ub_inp = None
        nb_samples = 1024
        self.net.eval()
        nb_inp = domain.size(0)

        start = time.time()
        last_improvement = start

        if use_cuda:
            self.net.cuda()
            domain = domain.cuda()


        while (time.time() - start < timeout) and (time.time() - last_improvement < noprogress_timeout):
            nb_starts += nb_samples
            if use_cuda:
                rand_samples = torch.cuda.FloatTensor(nb_samples, nb_inp)
            else:
                rand_samples = torch.FloatTensor(nb_samples, nb_inp)
            rand_samples.uniform_(0, 1)

            domain_lb = domain.select(1, 0).contiguous()
            domain_ub = domain.select(1, 1).contiguous()
            domain_width = domain_ub - domain_lb

            batch_domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
            batch_domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)
            inps = batch_domain_lb + batch_domain_width * rand_samples

            batch_ub = float('inf')
            while True:
                prev_batch_best = batch_ub

                var_inps = Variable(inps, requires_grad=True)
                out = self.net(var_inps)

                batch_ub = out.data.min()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    val, idx = out.data.min(dim=0)
                    best_ub_inp = inps[idx[0]].clone()
                    last_improvement = time.time()

                if batch_ub < prev_batch_best:
                    all_samp_sum = out.sum() / nb_samples
                    all_samp_sum.backward()
                    grad = var_inps.grad.data

                    max_grad, _ = grad.max(dim=0)
                    min_grad, _ = grad.min(dim=0)
                    grad_diff = max_grad - min_grad

                    lr = 1e-2 * domain_width / grad_diff
                    min_lr = lr.min()

                    step = -min_lr*grad
                    inps += step

                    inps = torch.max(inps, domain_lb.unsqueeze(0))
                    inps = torch.min(inps, domain_ub.unsqueeze(0))
                else:
                    break
            if early_stop and (best_ub < -1e-2):
                break
        return best_ub < 0, (best_ub_inp, best_ub), nb_starts

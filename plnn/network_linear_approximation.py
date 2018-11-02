import gurobipy as grb
import math
import torch

from plnn.modules import View
from torch import nn

class LinearizedNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)
        # Skip all gradient computation for the weights of the Net
        for param in self.net.parameters():
            param.requires_grad = False

    def remove_maxpools(self, domain):
        from plnn.model import reluify_maxpool, simplify_network
        if any(map(lambda x: type(x) is nn.MaxPool1d, self.layers)):
            new_layers = simplify_network(reluify_maxpool(self.layers, domain))
            self.layers = new_layers


    def get_upper_bound_random(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 1024
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        with torch.no_grad():
            inps = domain_lb + domain_width * rand_samples
            outs = self.net(inps)

            upper_bound, idx = torch.min(outs, dim=0)

            upper_bound = upper_bound[0].item()
            ub_point = inps[idx].squeeze()

        return ub_point, upper_bound

    def get_upper_bound_pgd(self, domain):
        '''
        Compute an upper bound of the minimum of the network on `domain`

        Any feasible point is a valid upper bound on the minimum so we will
        perform some random testing.
        '''
        nb_samples = 2056
        torch.set_num_threads(1)
        nb_inp = domain.size(0)
        # Not a great way of sampling but this will be good enough
        # We want to get rows that are >= 0
        rand_samples = torch.Tensor(nb_samples, nb_inp)
        rand_samples.uniform_(0, 1)

        best_ub = float('inf')
        best_ub_inp = None

        domain_lb = domain.select(1, 0).contiguous()
        domain_ub = domain.select(1, 1).contiguous()
        domain_width = domain_ub - domain_lb

        domain_lb = domain_lb.view(1, nb_inp).expand(nb_samples, nb_inp)
        domain_width = domain_width.view(1, nb_inp).expand(nb_samples, nb_inp)

        inps = (domain_lb + domain_width * rand_samples)

        with torch.enable_grad():
            batch_ub = float('inf')
            for i in range(1000):
                prev_batch_best = batch_ub

                self.net.zero_grad()
                if inps.grad is not None:
                    inps.grad.zero_()
                inps = inps.detach().requires_grad_()
                out = self.net(inps)

                batch_ub = out.min().item()
                if batch_ub < best_ub:
                    best_ub = batch_ub
                    # print(f"New best lb: {best_lb}")
                    _, idx = out.min(dim=0)
                    best_ub_inp = inps[idx[0]]

                if batch_ub >= prev_batch_best:
                    break

                all_samp_sum = out.sum() / nb_samples
                all_samp_sum.backward()
                grad = inps.grad

                max_grad, _ = grad.max(dim=0)
                min_grad, _ = grad.min(dim=0)
                grad_diff = max_grad - min_grad

                lr = 1e-2 * domain_width / grad_diff
                min_lr = lr.min()

                step = -min_lr*grad
                inps = inps + step

                inps = torch.max(inps, domain_lb)
                inps = torch.min(inps, domain_ub)

        return best_ub_inp, best_ub

    get_upper_bound = get_upper_bound_random

    def get_lower_bound(self, domain):
        '''
        Update the linear approximation for `domain` of the network and use it
        to compute a lower bound on the minimum of the output.

        domain: Tensor containing in each row the lower and upper bound for
                the corresponding dimension
        '''
        self.define_linear_approximation(domain)
        return self.compute_lower_bound(domain)

    def compute_lower_bound(self, domain):
        '''
        Compute a lower bound of the function on `domain`

        Note that this doesn't change the approximation that is made to tailor
        it to `domain`, which would lead to a better approximation.

        domain: Tensor containing in each row the lower and upper bound for the
                corresponding dimension.
        '''
        # We will first setup the appropriate bounds for the elements of the
        # input
        for var_idx, inp_var in enumerate(self.gurobi_vars[0]):
            inp_var.lb = domain[var_idx, 0]
            inp_var.ub = domain[var_idx, 1]

        # We will make sure that the objective function is properly set up
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)

        # We will now compute the requested lower bound
        self.model.update()
        self.model.optimize()
        assert self.model.status == 2, "LP wasn't optimally solved"

        return self.gurobi_vars[-1][0].X

    def define_linear_approximation(self, input_domain):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        '''
        self.lower_bounds = []
        self.upper_bounds = []
        self.gurobi_vars = []
        # These three are nested lists. Each of their elements will itself be a
        # list of the neurons after a layer.

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)

        ## Do the input layer, which is a special case
        inp_lb = []
        inp_ub = []
        inp_gurobi_vars = []
        for dim, (lb, ub) in enumerate(input_domain):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
            inp_lb.append(lb)
            inp_ub.append(ub)
        self.model.update()

        self.lower_bounds.append(inp_lb)
        self.upper_bounds.append(inp_ub)
        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        for layer in self.layers:
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                for neuron_idx in range(layer.weight.size(0)):
                    ub = layer.bias[neuron_idx].item()
                    lb = layer.bias[neuron_idx].item()
                    lin_expr = layer.bias[neuron_idx].item()
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight[neuron_idx, prev_neuron_idx].item()
                        if coeff >= 0:
                            ub += coeff*self.upper_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.lower_bounds[-1][prev_neuron_idx]
                        else:
                            ub += coeff*self.lower_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.upper_bounds[-1][prev_neuron_idx]
                        lin_expr += coeff * self.gurobi_vars[-1][prev_neuron_idx]
                    v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    self.model.setObjective(v, grb.GRB.MINIMIZE)
                    self.model.optimize()
                    assert self.model.status == 2, "LP wasn't optimally solved"
                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    self.model.setObjective(v, grb.GRB.MAXIMIZE)
                    self.model.update()
                    self.model.reset()
                    self.model.optimize()
                    assert self.model.status == 2, "LP wasn't optimally solved"
                    ub = v.X
                    v.ub = ub

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == nn.ReLU:
                for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                    pre_lb = self.lower_bounds[-1][neuron_idx]
                    pre_ub = self.upper_bounds[-1][neuron_idx]

                    v = self.model.addVar(lb=max(0, pre_lb),
                                          ub=max(0, pre_ub),
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'ReLU{layer_idx}_{neuron_idx}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        self.model.addConstr(v == pre_var)
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        self.model.addConstr(v >= pre_var)

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        self.model.addConstr(v <= slope * pre_var + bias)

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    lb = max(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                    ub = max(self.upper_bounds[-1][pre_start_idx:pre_window_end])

                    neuron_idx = pre_start_idx // stride

                    v = self.model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                          name=f'Maxpool{layer_idx}_{neuron_idx}')
                    all_pre_var = 0
                    for pre_var in self.gurobi_vars[-1][pre_start_idx:pre_window_end]:
                        self.model.addConstr(v >= pre_var)
                        all_pre_var += pre_var
                    all_lb = sum(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                    max_pre_lb = lb
                    self.model.addConstr(all_pre_var >= v + all_lb - max_pre_lb)

                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError

            self.lower_bounds.append(new_layer_lb)
            self.upper_bounds.append(new_layer_ub)
            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        self.model.update()


class InfeasibleMaskException(Exception):
    pass

class AssumptionLinearizedNetwork(LinearizedNetwork):

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        super(AssumptionLinearizedNetwork, self).__init__(layers)

    def get_initial_mask(self):
        '''
        Get a mask where all the non-linearities are unfixed.
        '''
        mask = []
        current_feat_size = -1
        for layer in self.layers:
            if type(layer) is nn.Linear:
                current_feat_size = layer.out_features
            elif type(layer) is nn.ReLU:
                layer_mask = [-1 for _ in range(current_feat_size)]
                mask.append(layer_mask)
            elif type(layer) is nn.MaxPool1d:
                nb_choices = layer.kernel_size
                nb_outputs = math.floor((current_feat_size + 2 * layer.padding
                                        - layer.dilation * (nb_choices-1) - 1)/layer.stride) + 1
                layer_mask = [-1 for _ in range(nb_outputs)]
                mask.append(layer_mask)
                current_feat_size = nb_outputs
            elif type(layer) is View:
                current_feat_size = layer.out_shape[-1]
        return mask

    def get_lower_bound(self, domain, relu_mask):
        try:
            new_mask = self.define_linear_approximation(domain, relu_mask)
            return self.compute_lower_bound(domain), new_mask
        except InfeasibleMaskException:
            # The model is infeasible, so this mask is wrong.
            # We just return an infinite lower bound
            return float('inf'), relu_mask

    def check_optimization_success(self):
        if self.model.status == 2:
            # Optimization successful, nothing to complain about
            pass
        elif self.model.status == 3:
            # The model is infeasible. We have made incompatible
            # assumptions, so this subdomain doesn't exist.
            raise InfeasibleMaskException()
        else:
            raise NotImplementedError

    def define_linear_approximation(self, input_domain, relu_mask):
        '''
        input_domain: Tensor containing in each row the lower and upper bound
                      for the corresponding dimension
        relu_mask: Indication of what ReLU we have already made assumption on.

        Returns a new version of relu_mask, udpated with any ReLU that can be inferred
        '''
        new_relu_mask = []

        self.lower_bounds = []
        self.upper_bounds = []
        self.gurobi_vars = []

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)

        ## Do the input layer, which is a special case
        inp_lb = []
        inp_ub = []
        inp_gurobi_vars = []
        for dim, (lb, ub) in enumerate(input_domain):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
            inp_lb.append(lb)
            inp_ub.append(ub)
        self.model.update()

        self.lower_bounds.append(inp_lb)
        self.upper_bounds.append(inp_ub)
        self.gurobi_vars.append(inp_gurobi_vars)

        ## Do the other layers, computing for each of the neuron, its upper
        ## bound and lower bound
        layer_idx = 1
        mask_idx = 0
        for layer in self.layers:
            new_layer_lb = []
            new_layer_ub = []
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                for neuron_idx in range(layer.weight.size(0)):
                    ub = layer.bias[neuron_idx].item()
                    lb = layer.bias[neuron_idx].item()
                    lin_expr = layer.bias[neuron_idx].item()
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight[neuron_idx, prev_neuron_idx].item()
                        if coeff >= 0:
                            ub += coeff*self.upper_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.lower_bounds[-1][prev_neuron_idx]
                        else:
                            ub += coeff*self.lower_bounds[-1][prev_neuron_idx]
                            lb += coeff*self.upper_bounds[-1][prev_neuron_idx]
                        lin_expr += coeff * self.gurobi_vars[-1][prev_neuron_idx]
                    v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lay{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    self.model.setObjective(v, grb.GRB.MINIMIZE)
                    self.model.optimize()
                    self.check_optimization_success()

                    # We have computed a lower bound
                    lb = v.X
                    v.lb = lb

                    # Let's now compute an upper bound
                    self.model.setObjective(v, grb.GRB.MAXIMIZE)
                    self.model.update()
                    self.model.reset()
                    self.model.optimize()
                    self.check_optimization_success()
                    ub = v.X
                    v.ub = ub

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == nn.ReLU:
                layer_mask = relu_mask[mask_idx]
                new_layer_mask = []
                for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                    mask_decision = layer_mask[neuron_idx]
                    pre_lb = self.lower_bounds[-1][neuron_idx]
                    pre_ub = self.upper_bounds[-1][neuron_idx]

                    if mask_decision == 0:
                        # The input should always be negative so if it isn't
                        # implied, we enforce it
                        if pre_ub > 0:
                            pre_ub = 0
                            assert pre_lb <= 0, "Lower bound isn't smaller than Upper bound"
                            self.model.addConstr(pre_var <= pre_ub)
                    elif mask_decision == 1:
                        # The input should always be positive so if it isn't
                        # implied, we enforce it
                        if pre_lb < 0:
                            pre_lb = 0
                            assert pre_ub >= 0, "Upper bound isn't bigger than Lower bound"
                            self.model.addConstr(pre_var >= pre_lb)
                    elif mask_decision == -1:
                        # Nothing imposed, just pass through normally
                        pass
                    else:
                        raise Exception("Unknown content in the mask.")

                    v = self.model.addVar(lb=max(0, pre_lb),
                                          ub=max(0, pre_ub),
                                          obj=0,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'ReLU{layer_idx}_{neuron_idx}')
                    if pre_lb >= 0 and pre_ub >= 0:
                        # The ReLU is always passing
                        self.model.addConstr(v == pre_var)
                        lb = pre_lb
                        ub = pre_ub
                    elif pre_lb <= 0 and pre_ub <= 0:
                        lb = 0
                        ub = 0
                        # No need to add an additional constraint that v==0
                        # because this will be covered by the bounds we set on
                        # the value of v.
                    else:
                        lb = 0
                        ub = pre_ub
                        self.model.addConstr(v >= pre_var)

                        slope = pre_ub / (pre_ub - pre_lb)
                        bias = - pre_lb * slope
                        self.model.addConstr(v <= slope * pre_var + bias)

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)

                    if pre_lb >= 0:
                        new_layer_mask.append(1)
                    elif pre_ub <= 0:
                        new_layer_mask.append(0)
                    else:
                        new_layer_mask.append(-1)
                mask_idx += 1
                new_relu_mask.append(new_layer_mask)
            elif type(layer) == nn.MaxPool1d:
                layer_mask = relu_mask[mask_idx]
                new_layer_mask = []
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size
                out_idx = 0
                while pre_window_end <= nb_pre:
                    neuron_idx = pre_start_idx // stride
                    if layer_mask[out_idx] == -1:
                        lb = max(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                        ub = max(self.upper_bounds[-1][pre_start_idx:pre_window_end])

                        v = self.model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                              name=f'Maxpool{layer_idx}_{neuron_idx}')
                        all_pre_var = 0
                        for pre_var in self.gurobi_vars[-1][pre_start_idx:pre_window_end]:
                            self.model.addConstr(v >= pre_var)
                            all_pre_var += pre_var
                        all_lb = sum(self.lower_bounds[-1][pre_start_idx:pre_window_end])
                        max_pre_lb = lb
                        self.model.addConstr(all_pre_var >= v + all_lb - max_pre_lb)

                        # Figure out if the selected out is implied. This
                        # implication is found when the highest lower bound is
                        # greater than the upper bound of all the other piece.
                        piece_dominated = [lb > inp_ub
                                           for inp_ub in self.upper_bounds[-1][pre_start_idx:pre_window_end]]
                        if sum(piece_dominated) == len(piece_dominated)-1:
                            # There is domination! (The -1 comes from the fact
                            # that the lower bound can't dominate itself)
                            chosen = piece_dominated.index(False)
                            new_layer_mask.append(chosen)
                        else:
                            new_layer_mask.append(-1)
                    else:
                        out_chosen = layer_mask[out_idx]
                        pre_var = self.gurobi_vars[-1][pre_start_idx + out_chosen]
                        lb = self.lower_bounds[-1][pre_start_idx + out_chosen]
                        ub = self.upper_bounds[-1][pre_start_idx + out_chosen]

                        v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                              vtype=grb.GRB.CONTINUOUS,
                                              name=f'Maxpool{layer_idx}_{neuron_idx}')
                        # Add constraint that the one indicated by the mask is
                        # selected
                        self.model.addConstr(v == pre_var)
                        # Add constraints that is indeed the one that should be
                        # selected
                        for other_idx, other_pre_var in enumerate(self.gurobi_vars[-1][pre_start_idx:pre_window_end]):
                            if other_idx != out_chosen:
                                self.model.addConstr(other_pre_var <= pre_var)
                        new_layer_mask.append(out_chosen)
                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                    out_idx += 1

                    new_layer_lb.append(lb)
                    new_layer_ub.append(ub)
                    new_layer_gurobi_vars.append(v)
                mask_idx += 1
                new_relu_mask.append(new_layer_mask)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError

            self.lower_bounds.append(new_layer_lb)
            self.upper_bounds.append(new_layer_ub)
            self.gurobi_vars.append(new_layer_gurobi_vars)

            layer_idx += 1

        # Assert that this is as expected a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # Check that the model is feasible
        self.model.update()
        self.model.setObjective(self.gurobi_vars[-1][0], grb.GRB.MINIMIZE)
        self.model.optimize()
        self.check_optimization_success()
        return new_relu_mask


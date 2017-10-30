import gurobipy as grb
import torch

from torch import nn
from plnn.modules import View
from plnn.network_linear_approximation import LinearizedNetwork

class MIPNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

        # Initialize a LinearizedNetwork object to determine the lower and
        # upper bounds at each layer.
        self.lin_net = LinearizedNetwork(layers)

    def solve(self, inp_domain):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        '''
        # First use define_linear_approximation from LinearizedNetwork to
        # compute upper and lower bounds to be able to define Ms
        self.lin_net.define_linear_approximation(inp_domain)

        self.lower_bounds = self.lin_net.lower_bounds
        self.upper_bounds = self.lin_net.upper_bounds
        self.gurobi_vars = []

        self.model = grb.Model()
        self.model.setParam('OutputFlag', False)
        self.model.setParam('Threads', 1)

        # First add the input variables as Gurobi variables.
        inp_gurobi_vars = []
        for dim, (lb, ub) in enumerate(inp_domain):
            v = self.model.addVar(lb=lb, ub=ub, obj=0,
                                  vtype=grb.GRB.CONTINUOUS,
                                  name=f'inp_{dim}')
            inp_gurobi_vars.append(v)

        self.gurobi_vars.append(inp_gurobi_vars)
        self.model.update()

        layer_idx = 1
        for layer in self.layers:
            new_layer_gurobi_vars = []
            if type(layer) is nn.Linear:
                for neuron_idx in range(layer.weight.size(0)):

                    lin_expr = layer.bias.data[neuron_idx]
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight.data[neuron_idx, prev_neuron_idx]
                        lin_expr += coeff * self.gurobi_vars[-1][prev_neuron_idx]
                    v = self.model.addVar(lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name=f'lin_v_{layer_idx}_{neuron_idx}')
                    self.model.addConstr(v == lin_expr)
                    self.model.update()

                    # We are now done with this neuron.
                    new_layer_gurobi_vars.append(v)

            elif type(layer) == nn.ReLU:

                for neuron_idx, pre_var in enumerate(self.gurobi_vars[-1]):
                    pre_lb = self.lower_bounds[layer_idx-1][neuron_idx]
                    pre_ub = self.upper_bounds[layer_idx-1][neuron_idx]

                    # Use the constraints specified by
                    # Maximum Resilience of Artificial Neural Networks paper.
                    # MIP formulation of ReLU:
                    #
                    # x = max(pre_var, 0)
                    #
                    # Introduce binary variable b, such that:
                    # b = 1 if in is the maximum value, 0 otherwise
                    #
                    # Introduce a continuous variable M, such that -M <= pre_var <= M:
                    #
                    # We know the lower (pre_lb) and upper bounds (pre_ub) for pre_var
                    # We can thus write the following:
                    # M = max(-pre_lb, pre_ub)
                    #
                    # MIP must then satisfy the following constraints:
                    # Constr_2a: x >= 0
                    # Constr_2b: x >= pre_var
                    # Constr_3a: pre_var - b*M <= 0
                    # Constr_3b: pre_var + (1-b)*M >= 0
                    # Constr_4a: x <= pre_var + (1-b)*M
                    # Constr_4b: x <= b*M

                    M = max(-pre_lb, pre_ub)
                    x = self.model.addVar(lb=0,
                                          ub=grb.GRB.INFINITY,
                                          vtype=grb.GRB.CONTINUOUS,
                                          name = f'RelU_x_{layer_idx}_{neuron_idx}')
                    b = self.model.addVar(vtype=grb.GRB.BINARY,
                                          name= f'ReLU_b_{layer_idx}_{neuron_idx}')

                    self.model.addConstr(x >= 0, f'constr_{layer_idx}_{neuron_idx}_c2a')
                    self.model.addConstr(x >= pre_var, f'constr_{layer_idx}_{neuron_idx}_c2b')
                    self.model.addConstr(pre_var - b*M <= 0, f'constr_{layer_idx}_{neuron_idx}_c3a')
                    self.model.addConstr(pre_var + (1-b)*M >= 0, f'constr_{layer_idx}_{neuron_idx}_c3b')
                    self.model.addConstr(x <= pre_var + (1-b)*M , f'constr_{layer_idx}_{neuron_idx}_c4a')
                    self.model.addConstr(x <= b*M , f'constr_{layer_idx}_{neuron_idx}_c4b')

                    self.model.update()

                    new_layer_gurobi_vars.append(x)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    ub_max = max(self.upper_bounds[layer_idx-1][pre_start_idx:pre_window_end])
                    window_bin_vars = []
                    neuron_idx = pre_start_idx % stride
                    v = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=-grb.GRB.INFINITY,
                                          ub=grb.GRB.INFINITY,
                                          name=f'MaxPool_out_{layer_idx}_{neuron_idx}')
                    for pre_var_idx, pre_var in enumerate(self.gurobi_vars[-1][pre_start_idx:pre_window_end]):
                        lb = self.lower_bounds[layer_idx-1][pre_start_idx + pre_var_idx]
                        b = self.model.addVar(vtype=grb.GRB.BINARY,
                                              name= f'MaxPool_b_{layer_idx}_{neuron_idx}_{pre_var_idx}')
                        # MIP formulation of max pooling:
                        #
                        # y = max(x_1, x_2, ..., x_n)
                        #
                        # Introduce binary variables d_1, d_2, ..., d_n:
                        # d_i = i if x_i is the maximum value, 0 otherwise
                        #
                        # We know the lower (l_i) and upper bounds (u_i) for x_i
                        #
                        # Denote the maximum of the upper_bounds of all inputs x_i as u_max
                        #
                        # MIP must then satisfy the following constraints:
                        #
                        # Constr_1: l_i <= x_i <= u_i
                        # Constr_2: y >= x_i
                        # Constr_3: y <= x_i + (u_max - l_i)*(1 - d_i)
                        # Constr_4: sum(d_1, d_2, ..., d_n) = 1

                        # Constr_1 is already satisfied due to the implementation of LinearizedNetworks.
                        # Constr_2
                        self.model.addConstr(v >= pre_var)
                        # Constr_3
                        self.model.addConstr(v <= pre_var + (ub_max - lb)*(1-b))

                        window_bin_vars.append(b)
                    # Constr_4
                    self.model.addConstr(sum(window_bin_vars) == 1)
                    self.model.update()
                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                    new_layer_gurobi_vars.append(v)
            elif type(layer) == View:
                continue
            else:
                raise NotImplementedError

            self.gurobi_vars.append(new_layer_gurobi_vars)
            layer_idx += 1
        # Assert that this is as expected: a network with a single output
        assert len(self.gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

        # Add the final constraint that the output must be less than or equal
        # to zero.
        self.model.addConstr(self.gurobi_vars[-1][-1] <= 0)

        # Optimize the model.
        self.model.update()
        self.model.setObjective(0, grb.GRB.MAXIMIZE)
        self.model.optimize()

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution
            return (False, None)
        else:
            # There is a feasible solution. Return the feasible solution as well.
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            inp = torch.Tensor(len_inp)
            for idx, var in enumerate(self.gurobi_vars[0]):
                inp[idx] = var.x

            return (True, inp)

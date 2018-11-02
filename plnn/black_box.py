import gurobipy as grb
import torch

from torch import nn
from plnn.modules import View
from plnn.network_linear_approximation import LinearizedNetwork

class BlackBoxNetwork:

    def __init__(self, layers):
        '''
        layers: A list of Pytorch layers containing only Linear/ReLU/MaxPools
        '''
        self.layers = layers
        self.net = nn.Sequential(*layers)

    def solve(self, inp_domain):
        '''
        inp_domain: Tensor containing in each row the lower and upper bound
                    for the corresponding dimension

        Returns:
        sat     : boolean indicating whether the MIP is satisfiable.
        solution: Feasible point if the MIP is satisfiable,
                  None otherwise.
        '''
        if self.check_obj_value_callback:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIP:
                    best_bound = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
                    if best_bound > 0:
                        model.terminate()

                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

                if where == grb.GRB.Callback.MIPSOL:
                    obj = model.cbGet(grb.GRB.Callback.MIPSOL_OBJ)
                    if obj < 0:
                        # Does it have a chance at being a valid
                        # counter-example?

                        # Check it with the network
                        input_vals = model.cbGetSolution(self.gurobi_vars[0])

                        with torch.no_grad():
                            inps = torch.Tensor(input_vals).view(1, -1)
                            out = self.net(inps).item()

                        if out < 0:
                            model.terminate()
        else:
            def early_stop_cb(model, where):
                if where == grb.GRB.Callback.MIPNODE:
                    nodeCount = model.cbGet(grb.GRB.Callback.MIPNODE_NODCNT)
                    if (nodeCount % 100) == 0:
                        print(f"Running Nb states visited: {nodeCount}")

        self.model.optimize(early_stop_cb)
        nb_visited_states = self.model.nodeCount

        if self.model.status is grb.GRB.INFEASIBLE:
            # Infeasible: No solution
            return (False, None, nb_visited_states)
        elif self.model.status is grb.GRB.OPTIMAL:
            # There is a feasible solution. Return the feasible solution as well.
            len_inp = len(self.gurobi_vars[0])

            # Get the input that gives the feasible solution.
            inp = torch.Tensor(len_inp)
            for idx, var in enumerate(self.gurobi_vars[0]):
                inp[idx] = var.x
            optim_val = self.gurobi_vars[-1][-1].x

            return (optim_val < 0, (inp, optim_val), nb_visited_states)
        elif self.model.status is grb.GRB.INTERRUPTED:
            obj_bound = self.model.ObjBound

            if obj_bound > 0:
                return (False, None, nb_visited_states)
            else:
                # There is a feasible solution. Return the feasible solution as well.
                len_inp = len(self.gurobi_vars[0])

                # Get the input that gives the feasible solution.
                inp = torch.Tensor(len_inp)
                for idx, var in enumerate(self.gurobi_vars[0]):
                    inp[idx] = var.x
                optim_val = self.gurobi_vars[-1][-1].x
            return (optim_val < 0, (inp, optim_val), nb_visited_states)

    def setup_model(self, inp_domain, use_obj_function=False):
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

                    lin_expr = layer.bias[neuron_idx].item()
                    for prev_neuron_idx in range(layer.weight.size(1)):
                        coeff = layer.weight[neuron_idx, prev_neuron_idx].item()
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
                    new_var = self.model.addVar(lb=0, ub=grb.GRB.INFINITY,
                                                vtype=grb.GRB.CONTINUOUS,
                                                name=f'ReLU_x_{layer_idx}_{neuron_idx}')
                    self.model.addGenConstrMax(new_var, [pre_var],
                                               constant=0,
                                               name=f"constr_{layer_idx}_{neuron_idx}")
                    self.model.update()

                    new_layer_gurobi_vars.append(new_var)
            elif type(layer) == nn.MaxPool1d:
                assert layer.padding == 0, "Non supported Maxpool option"
                assert layer.dilation == 1, "Non supported MaxPool option"
                nb_pre = len(self.gurobi_vars[-1])
                window_size = layer.kernel_size
                stride = layer.stride

                pre_start_idx = 0
                pre_window_end = pre_start_idx + window_size

                while pre_window_end <= nb_pre:
                    window_bin_vars = []
                    neuron_idx = pre_start_idx // stride
                    new_var = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                                lb=-grb.GRB.INFINITY,
                                                ub=grb.GRB.INFINITY,
                                                name=f'MaxPool_out_{layer_idx}_{neuron_idx}')
                    pre_vars = self.gurobi_vars[-1][pre_start_idx:pre_window_end]
                    self.model.addGenConstrMax(new_var, pre_vars,
                                               name=f"Maxpool_constr_{layer_idx}_{neuron_idx}")
                    self.model.update()
                    pre_start_idx += stride
                    pre_window_end = pre_start_idx + window_size
                    new_layer_gurobi_vars.append(new_var)
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
        if not use_obj_function:
            self.model.addConstr(self.gurobi_vars[-1][-1] <= 0)
            self.model.setObjective(0, grb.GRB.MAXIMIZE)
            self.check_obj_value_callback = False
        else:
            self.model.setObjective(self.gurobi_vars[-1][-1], grb.GRB.MINIMIZE)
            self.check_obj_value_callback = True

        # Optimize the model.
        self.model.update()

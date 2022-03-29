import itertools

import numpy as np
import pyomo.kernel as pyo
import torch

import utils
from definitions import input_lower_bound, input_upper_bound


def min_and_max_output_layer(x_in_max, x_in_min, W, b):
    W_positive = np.maximum(W, 0)
    W_negative = np.minimum(W, 0)
    y_max = x_in_max @ W_positive.T + x_in_min @ W_negative.T + b.T
    y_min = x_in_max @ W_negative.T + x_in_min @ W_positive.T + b.T

    return y_max, y_min


class VerificationModel:
    """
    The reformulation of the neural network into a MILP program implemented as a class with multiple functions for
     performing the bound tightening and solving various verification or embedding problems.
    """

    def __init__(self, network, margin, n_threads: int = 1, relaxed=False, verbose=True):
        """

        :param network: The neural network
        :param margin: The upper and lower bound of the marginal region
        :param n_threads: The number of threads assigned to solving the optimisation problems
        :param relaxed: If the ReLUs are treated as binary (relaxed=False) or in the relaxed version as Real variables with bounds of [0,1]
        :param verbose: Display of print statements during the process (e.g., bound tightening)
        """

        # Data needed
        network.eval()
        self.network = network
        self.network_state_dict = network.state_dict()

        self.n_hidden_layers = network.n_hidden_layers
        self.hidden_layer_size = network.hidden_layer_size
        self.n_input_neurons = network.n_input_neurons
        self.n_output_neurons = network.n_output_neurons  # size output layer . our case it is 1
        self.relaxed_model = relaxed

        self.input_lower_bound = input_lower_bound
        self.input_upper_bound = input_upper_bound

        self.mean_in = self.network_state_dict[f'dense_layers.input_standardisation.mean'].data.numpy()
        self.standard_deviation_in = self.network_state_dict[
            f'dense_layers.input_standardisation.standard_deviation'].data.numpy()
        self.mean_out = self.network_state_dict[f'dense_layers.output_scaling.mean'].data.numpy()
        self.standard_deviation_out = self.network_state_dict[
            f'dense_layers.output_scaling.standard_deviation'].data.numpy()

        self.verbose = verbose
        model = pyo.block()
        model.dual = pyo.suffix(direction=pyo.suffix.IMPORT)

        # Parameters -----------------------------------------------
        # weights and biases
        weights = pyo.parameter_list()
        biases = pyo.parameter_list()
        for layer in range(self.n_hidden_layers):
            layer_weights = self.network_state_dict[f'dense_layers.dense_{layer}.weight'].data.numpy()
            layer_biases = self.network_state_dict[f'dense_layers.dense_{layer}.bias'].data.numpy()
            weights.append(pyo.parameter_list())
            biases.append(pyo.parameter(value=layer_biases))
            for neuron in range(layer_weights.shape[0]):
                weights[layer].append(pyo.parameter(value=layer_weights[neuron, :]))

        output_weights = self.network_state_dict[f'dense_layers.output_layer.weight'].data.numpy()
        output_biases = self.network_state_dict[f'dense_layers.output_layer.bias'].data.numpy()
        weights.append(pyo.parameter_list())
        biases.append(pyo.parameter(value=output_biases))
        for neuron in range(output_weights.shape[0]):
            weights[-1].append(pyo.parameter(value=output_weights[neuron, :]))

        model.weights = weights
        model.biases = biases

        # bounds on neuron values
        z_hat_min_initial = list()
        z_hat_max_initial = list()

        for layer in range(self.n_hidden_layers):
            z_hat_min_initial.append([-10000] * self.hidden_layer_size)
            z_hat_max_initial.append([10000] * self.hidden_layer_size)

        z_hat_max = pyo.parameter_list()
        z_hat_min = pyo.parameter_list()
        for z_hat_max_layer, z_hat_min_layer in zip(z_hat_max_initial, z_hat_min_initial):
            z_hat_max.append(pyo.parameter(value=z_hat_max_layer))
            z_hat_min.append(pyo.parameter(value=z_hat_min_layer))

        model.z_hat_max = z_hat_max
        model.z_hat_min = z_hat_min

        # prediction classification bound
        model.bound_unstable = pyo.parameter(value=margin[0])
        model.bound_stable = pyo.parameter(value=margin[1])

        # input variable bounds
        model.input_lower_bound = pyo.parameter(value=input_lower_bound[0, :].tolist())
        model.input_upper_bound = pyo.parameter(value=input_upper_bound[0, :].tolist())

        # Variables -----------------------------------------------
        model.epsilon_lower_bound = pyo.parameter(value=0.0)
        model.epsilon_upper_bound = pyo.parameter(value=1.0)
        model.epsilon = pyo.variable()

        # input variables
        model.input_scaled = pyo.variable_dict()
        for ii in range(self.n_input_neurons):
            model.input_scaled[ii] = pyo.variable(lb=model.input_lower_bound.value[ii],
                                                  ub=model.input_upper_bound.value[ii])

        model.input_reference_point = pyo.variable_list()
        for ii in range(self.n_input_neurons):
            model.input_reference_point.append(pyo.variable(lb=model.input_lower_bound.value[ii],
                                                            ub=model.input_upper_bound.value[ii],
                                                            value=(model.input_lower_bound.value[ii] +
                                                                   model.input_upper_bound.value[ii]) / 2,
                                                            fixed=True))

        # ReLUs (binary or continuous [0;1], selected by 'relaxed')
        model.ReLUs = pyo.variable_list()

        for layer in range(self.n_hidden_layers):
            model.ReLUs.append(pyo.variable_list())
            for neuron in range(self.hidden_layer_size):
                if relaxed:
                    model.ReLUs[layer].append(pyo.variable(lb=0.0, ub=1.0))
                else:
                    model.ReLUs[layer].append(pyo.variable(domain=pyo.Binary))

        # neuron value after activation
        model.z = pyo.variable_list()

        for layer in range(self.n_hidden_layers):
            model.z.append(pyo.variable_list())
            for neuron in range(self.hidden_layer_size):
                model.z[layer].append(pyo.variable(value=1))

        # neuron value before activation
        model.z_hat = pyo.variable_list()

        for layer in range(self.n_hidden_layers):
            model.z_hat.append(pyo.variable_list())
            for neuron in range(self.hidden_layer_size):
                model.z_hat[layer].append(pyo.variable(value=1))

        # bounds neuron value before activation
        model.z_hat_max_bound_tightening = pyo.variable_list()
        model.z_hat_min_bound_tightening = pyo.variable_list()

        for layer in range(self.n_hidden_layers):
            model.z_hat.append(pyo.variable_list())
            model.z_hat_max_bound_tightening.append(pyo.variable_list())
            model.z_hat_min_bound_tightening.append(pyo.variable_list())
            for neuron in range(self.hidden_layer_size):
                model.z_hat[layer].append(pyo.variable(value=1))
                model.z_hat_max_bound_tightening[layer].append(pyo.variable(value=100000))
                model.z_hat_min_bound_tightening[layer].append(pyo.variable(value=-100000))

        # model prediction
        model.prediction = pyo.variable_list()
        for output_neuron in range(self.n_output_neurons):
            model.prediction.append(pyo.variable())

        # Constraints -----------------------------------------------
        # Dense layer constraints (input layer, hidden layers, output layer)
        model.dense_layers = pyo.constraint_list()

        model.dense_layers.append(pyo.constraint_list())
        for neuron in range(self.hidden_layer_size):
            Lhs = model.z_hat[0][neuron] - (sum(
                model.weights[0][neuron].value[k] * (model.input_scaled[k] - self.mean_in[k]) / (
                        self.standard_deviation_in[k] + 1e-8)
                for k in range(self.n_input_neurons)) +
                                            model.biases[0].value[neuron])
            model.dense_layers[0].append(pyo.constraint(body=Lhs, rhs=0))

        for layer in range(self.n_hidden_layers)[:-1]:
            model.dense_layers.append(pyo.constraint_list())
            for neuron in range(self.hidden_layer_size):
                Lhs = model.z_hat[layer + 1][neuron] - (sum(
                    model.weights[layer + 1][neuron].value[k] * model.z[layer][k] for k in
                    range(self.hidden_layer_size)) +
                                                        model.biases[layer + 1].value[neuron])
                model.dense_layers[layer + 1].append(pyo.constraint(body=Lhs, rhs=0))

        model.dense_layers.append(pyo.constraint_list())
        for neuron in range(self.n_output_neurons):
            Lhs = model.prediction[neuron] - ((sum(
                model.weights[-1][neuron].value[k] * model.z[-1][k] for k in range(self.hidden_layer_size)) +
                                               model.biases[-1].value[neuron]) * self.standard_deviation_out[neuron] +
                                              self.mean_out[neuron])
            model.dense_layers[-1].append(pyo.constraint(body=Lhs, rhs=0))

        # ReLU constraints (4 for each neuron)
        model.relu_constraints = pyo.constraint_list()

        for layer in range(self.n_hidden_layers):
            model.relu_constraints.append(pyo.constraint_list())
            for neuron in range(self.hidden_layer_size):
                model.relu_constraints[layer].append(pyo.constraint_dict())

                Lhs = model.z[layer][neuron] - model.z_hat[layer][neuron] + model.z_hat_min[layer].value[neuron] * (
                        1 - model.ReLUs[layer][neuron])
                model.relu_constraints[layer][neuron][0] = pyo.constraint(expr=Lhs <= 0)

                Lhs = model.z[layer][neuron] - model.z_hat[layer][neuron]
                model.relu_constraints[layer][neuron][1] = pyo.constraint(expr=Lhs >= 0)

                Lhs = model.z[layer][neuron] - model.z_hat_max[layer].value[neuron] * model.ReLUs[layer][neuron]
                model.relu_constraints[layer][neuron][2] = pyo.constraint(expr=Lhs <= 0)

                Lhs = model.z[layer][neuron]
                model.relu_constraints[layer][neuron][3] = pyo.constraint(expr=Lhs >= 0)

        # Reference point constraints
        model.reference_point_constraint = pyo.constraint_list()

        for ii in range(self.n_input_neurons):
            Lhs = (model.input_scaled[ii] - model.input_reference_point[ii])
            Rhs = (model.input_upper_bound.value[ii] - model.input_lower_bound.value[ii]) * model.epsilon
            model.reference_point_constraint.append(pyo.constraint(expr=Lhs <= Rhs))
            model.reference_point_constraint.append(pyo.constraint(expr=-Lhs <= Rhs))

        # Prediction classification constraint
        model.constraint_prediction_unstable = pyo.constraint(expr=model.prediction[0] >= model.bound_unstable)
        model.constraint_prediction_stable = pyo.constraint(expr=model.prediction[0] <= model.bound_stable)

        # epsilon bounds constraint
        model.epsilon_lower_bound_constraint = pyo.constraint(expr=model.epsilon >= model.epsilon_lower_bound)
        model.epsilon_upper_bound_constraint = pyo.constraint(expr=model.epsilon <= model.epsilon_upper_bound)

        # Bound tightening constraints, by default deactivated
        model.bound_tightening_constraints = pyo.constraint_list()
        model.bound_tightening_constraints_z_hat_max = pyo.constraint_list()
        model.bound_tightening_constraints_z_hat_min = pyo.constraint_list()

        for layer in range(self.n_hidden_layers):
            model.bound_tightening_constraints.append(pyo.constraint_list())
            model.bound_tightening_constraints_z_hat_max.append(pyo.constraint_list())
            model.bound_tightening_constraints_z_hat_min.append(pyo.constraint_list())
            for neuron in range(self.hidden_layer_size):
                model.bound_tightening_constraints[layer].append(pyo.constraint_dict())

                Lhs = model.z_hat[layer][neuron] - model.epsilon
                model.bound_tightening_constraints[layer][neuron][0] = pyo.constraint(body=Lhs, rhs=0)

                Lhs = model.z_hat[layer][neuron] + model.epsilon
                model.bound_tightening_constraints[layer][neuron][1] = pyo.constraint(body=Lhs, rhs=0)

                model.bound_tightening_constraints_z_hat_max[layer].append(
                    pyo.constraint(expr=model.epsilon >= -model.z_hat_max_bound_tightening[layer][neuron]))
                model.bound_tightening_constraints_z_hat_min[layer].append(
                    pyo.constraint(expr=model.epsilon >= model.z_hat_min_bound_tightening[layer][neuron]))

                model.bound_tightening_constraints[layer][neuron][0].deactivate()
                model.bound_tightening_constraints[layer][neuron][1].deactivate()
                model.bound_tightening_constraints_z_hat_max[layer][neuron].deactivate()
                model.bound_tightening_constraints_z_hat_min[layer][neuron].deactivate()

        # Objectives
        model.objective_minimise_epsilon = pyo.objective(model.epsilon, sense=pyo.minimize)

        self.solver = pyo.SolverFactory('gurobi')
        self.solver.options['threads'] = n_threads
        self.pyo_model = model

    def predict(self, input_value):
        """
        Simple call using the neural network, helpful as a check for the optimisation solution.
        :param input_value: Data points, one per row
        :return: predictions (as np.array)
        """
        input_tensor = utils.ensure_tensor_array(input_array=input_value)
        with torch.no_grad():
            prediction = self.network(input_tensor)

        return utils.ensure_numpy_array(prediction)

    def set_ReLUs_as_binary(self):
        """
        Change the variable type of the ReLUs to binary for the exact (un-relaxed) model.
        """
        for layer in range(self.n_hidden_layers):
            for neuron in range(self.hidden_layer_size):
                self.pyo_model.ReLUs[layer][neuron].domain = pyo.Binary
        self.relaxed_model = False
        pass

    def set_ReLUs_as_relaxed(self):
        """
        Change the variable type of the ReLUs to UnitInterval for the relaxed model.
        """
        for layer in range(self.n_hidden_layers):
            for neuron in range(self.hidden_layer_size):
                self.pyo_model.ReLUs[layer][neuron].domain = pyo.UnitInterval
            pass
        self.relaxed_model = True
        pass

    def perform_analytical_bound_tightening(self):
        """
        Simple pass through the neural network to find smaller z_hat_max and z_hat_min
        :return:
        """
        input_upper_limit = (self.input_upper_bound - self.mean_in) / (self.standard_deviation_in + 1e-8)
        input_lower_limit = (self.input_lower_bound - self.mean_in) / (self.standard_deviation_in + 1e-8)

        weights = list()
        biases = list()
        for layer in range(self.n_hidden_layers):
            layer_weights = self.network_state_dict[f'dense_layers.dense_{layer}.weight'].data.numpy()
            layer_biases = self.network_state_dict[f'dense_layers.dense_{layer}.bias'].data.numpy().reshape((-1, 1))
            weights.append(layer_weights)
            biases.append(layer_biases)

        z_hat_max_new, z_hat_min_new = min_and_max_output_layer(input_upper_limit,
                                                                input_lower_limit,
                                                                weights[0],
                                                                biases[0])

        self.pyo_model.z_hat_max[0].value = z_hat_max_new[0, :].tolist()
        self.pyo_model.z_hat_min[0].value = z_hat_min_new[0, :].tolist()

        for layer in range(self.n_hidden_layers - 1):
            z_max = np.maximum(self.pyo_model.z_hat_max[layer].value, 0)
            z_min = np.maximum(self.pyo_model.z_hat_min[layer].value, 0)

            z_hat_max_new, z_hat_min_new = min_and_max_output_layer(z_max,
                                                                    z_min,
                                                                    weights[layer + 1],
                                                                    biases[layer + 1])

            self.pyo_model.z_hat_max[layer + 1].value = z_hat_max_new[0, :].tolist()
            self.pyo_model.z_hat_min[layer + 1].value = z_hat_min_new[0, :].tolist()

        pass

    def perform_optimisation_bound_tightening(self, layer_index):
        """
        Tighten the values of z_hat_max and z_hat_min for a given layer.
        :param layer_index: index of the layer, starting at 0
        """
        if self.verbose:
            print(f'Tightening of z_hat_min and z_hat max in layer {layer_index + 1}')
            print(f'Relaxed version: {self.relaxed_model}')

        for ii in range(len(self.pyo_model.reference_point_constraint)):
            self.pyo_model.reference_point_constraint[ii].deactivate()

        # Prediction classification constraint
        self.pyo_model.constraint_prediction_unstable.deactivate()
        self.pyo_model.constraint_prediction_stable.deactivate()

        self.pyo_model.epsilon_lower_bound_constraint.deactivate()
        self.pyo_model.epsilon_upper_bound_constraint.deactivate()

        z_hat_min_old_layer = self.pyo_model.z_hat_min[layer_index].value
        z_hat_max_old_layer = self.pyo_model.z_hat_max[layer_index].value

        for neuron in range(self.hidden_layer_size):
            self.pyo_model.z_hat_min_bound_tightening[layer_index][neuron].fix(
                self.pyo_model.z_hat_min[layer_index].value[neuron])
            self.pyo_model.z_hat_max_bound_tightening[layer_index][neuron].fix(
                self.pyo_model.z_hat_max[layer_index].value[neuron])

        z_hat_min_new_layer = list()
        z_hat_max_new_layer = list()
        for neuron in range(self.hidden_layer_size):
            # tighter z_hat_min
            self.pyo_model.bound_tightening_constraints_z_hat_min[layer_index][neuron].activate()
            self.pyo_model.bound_tightening_constraints[layer_index][neuron][0].activate()

            self.solver.solve(self.pyo_model)
            z_hat_min_new_layer.append(self.pyo_model.z_hat[layer_index][neuron].value)

            self.pyo_model.bound_tightening_constraints[layer_index][neuron][0].deactivate()
            self.pyo_model.bound_tightening_constraints_z_hat_min[layer_index][neuron].deactivate()

            # tighter z_hat_max
            self.pyo_model.bound_tightening_constraints_z_hat_max[layer_index][neuron].activate()
            self.pyo_model.bound_tightening_constraints[layer_index][neuron][1].activate()

            self.solver.solve(self.pyo_model)
            z_hat_max_new_layer.append(self.pyo_model.z_hat[layer_index][neuron].value)

            self.pyo_model.bound_tightening_constraints[layer_index][neuron][1].deactivate()
            self.pyo_model.bound_tightening_constraints_z_hat_max[layer_index][neuron].deactivate()

        self.pyo_model.z_hat_min[layer_index].value = z_hat_min_new_layer
        self.pyo_model.z_hat_max[layer_index].value = z_hat_max_new_layer

        bound_reduction = np.mean(
            (np.array(z_hat_max_new_layer) - np.array(z_hat_min_new_layer)) / (
                    np.array(z_hat_max_old_layer) - np.array(z_hat_min_old_layer)))

        if self.verbose:
            print(f'Bound reduction layer {layer_index + 1}: {100 - bound_reduction * 100:.1f} %')

        for ii in range(len(self.pyo_model.reference_point_constraint)):
            self.pyo_model.reference_point_constraint[ii].activate()

        # Prediction classification constraint
        self.pyo_model.constraint_prediction_unstable.activate()
        self.pyo_model.constraint_prediction_stable.activate()

        pass

    def perform_full_bound_tightening(self):
        """
        Perform all possible bound tightening versions with increasing computational effort.
        """
        self.perform_analytical_bound_tightening()
        self.set_ReLUs_as_relaxed()
        for ii in range(1, self.n_hidden_layers):
            self.perform_optimisation_bound_tightening(ii)

        self.set_ReLUs_as_binary()
        for ii in range(1, self.n_hidden_layers):
            self.perform_optimisation_bound_tightening(ii)

        pass

    def solve_for_largest_region_around_reference_point(self, reference_point, set_point=None):
        """
        Find the largest region (measured in the unit-hypercube) around a reference point. If the optimisation shall
        only consider a subset of the dimensions, values of the reference points can be indicated to be set point and
        are thereby fixed in the optimisation problem.
        :param reference_point: A vector specifying where epsilon = 0
        :param set_point: A vector of binaries indicating which values should not be altered (set_point = True)
        :return: the verified radius epsilon, the resulting point from the optimisation and the corresponding prediction
        """

        if set_point is None:
            set_point = [False] * self.n_input_neurons

        for ii, reference_value in enumerate(reference_point):
            self.pyo_model.input_reference_point[ii].value = reference_value

        for ii, set_point_check in enumerate(set_point):
            if set_point_check:
                self.pyo_model.input_scaled[ii].value = reference_point[ii]
                self.pyo_model.input_scaled[ii].fix()

        reference_point_prediction = self.predict(input_value=reference_point)

        if reference_point_prediction <= self.pyo_model.bound_unstable.value:
            self.pyo_model.constraint_prediction_unstable.activate()
            self.pyo_model.constraint_prediction_stable.deactivate()
            self.solver.solve(self.pyo_model)

            verified_radius = self.pyo_model.epsilon.value
            resulting_point = np.array([[self.pyo_model.input_scaled[0].value,
                                         self.pyo_model.input_scaled[1].value,
                                         self.pyo_model.input_scaled[2].value,
                                         self.pyo_model.input_scaled[3].value]])

        elif reference_point_prediction >= self.pyo_model.bound_stable.value:
            self.pyo_model.constraint_prediction_unstable.deactivate()
            self.pyo_model.constraint_prediction_stable.activate()
            self.solver.solve(self.pyo_model)

            verified_radius = self.pyo_model.epsilon.value
            resulting_point = np.array([[self.pyo_model.input_scaled[0].value,
                                         self.pyo_model.input_scaled[1].value,
                                         self.pyo_model.input_scaled[2].value,
                                         self.pyo_model.input_scaled[3].value]])

        else:
            verified_radius = 0.0
            resulting_point = reference_point

        # TODO: Handle the no-solution case (the desired behaviour is verified across the entire domain)
        # free the input variables again
        for ii, set_point_check in enumerate(set_point):
            if set_point_check:
                self.pyo_model.input_scaled[ii].fixed = False

        resulting_point_prediction = self.predict(input_value=utils.ensure_tensor_array(resulting_point))

        return verified_radius, resulting_point, resulting_point_prediction

    def solve_for_maximum_control_parameters(self, reference_point):
        """
        Special case of 'solve_for_largest_region_around_reference_point' if a power set point is given
        """
        return self.solve_for_largest_region_around_reference_point(reference_point=reference_point,
                                                                    set_point=[True, True, False, False])

    def solve_for_maximum_power_parameters(self, reference_point):
        """
        Special case of 'solve_for_largest_region_around_reference_point' if a control set point is given
        """
        return self.solve_for_largest_region_around_reference_point(reference_point=reference_point,
                                                                    set_point=[False, False, True, True])

    def solve_regions_around_corner_points(self):
        """
        Special case of 'solve_for_largest_region_around_reference_point' if the reference points are the corner points
        of the domain hypercube
        """
        corner_points_unscaled = np.vstack(
            list(itertools.product(*zip([0] * self.n_input_neurons, [1] * self.n_input_neurons))))
        corner_points_scaled = utils.scale_array_with_input_bounds(corner_points_unscaled)

        results = [self.solve_for_largest_region_around_reference_point(corner_point) for corner_point in
                   corner_points_scaled]

        verified_radii = np.vstack([result_element[0] for result_element in results])
        resulting_point = np.vstack([result_element[1].reshape((1, -1)) for result_element in results])
        resulting_point_prediction = np.vstack([result_element[2].reshape((1, -1)) for result_element in results])

        corner_predictions = self.predict(input_value=corner_points_scaled)

        return corner_points_scaled, verified_radii, corner_predictions, resulting_point, resulting_point_prediction

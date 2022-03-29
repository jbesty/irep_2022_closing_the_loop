from collections import OrderedDict

import torch
from torch import nn


class Standardise(nn.Module):
    """
    Scale the input to the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Standardise, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input):
        return (input - self.mean) / (self.standard_deviation + self.eps)

    def set_standardisation(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.standard_deviation).shape[0] == standard_deviation.shape[0]:
            raise Exception('Standard deviation in standardisation contains elements equal to 0.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = nn.Parameter(data=standard_deviation, requires_grad=False)


class Scale(nn.Module):
    """
    Scale the input to the layer by mean and standard deviation.
    """

    def __init__(self, n_neurons):
        super(Scale, self).__init__()
        self.mean = nn.Parameter(data=torch.zeros(n_neurons), requires_grad=False)
        self.standard_deviation = nn.Parameter(data=torch.ones(n_neurons), requires_grad=False)
        self.eps = 1e-8

    def forward(self, input):
        return self.mean + input * self.standard_deviation

    def set_scaling(self, mean, standard_deviation):
        if not len(standard_deviation.shape) == 1 or not len(mean.shape) == 1:
            raise Exception('Input statistics are not 1-D tensors.')

        if not torch.nonzero(self.standard_deviation).shape[0] == standard_deviation.shape[0]:
            raise Exception('Standard deviation in scaling contains elements equal to 0.')

        self.mean = nn.Parameter(data=mean, requires_grad=False)
        self.standard_deviation = nn.Parameter(data=standard_deviation, requires_grad=False)


class NeuralNetwork(torch.nn.Module):
    """
    A simple multi-layer perceptron network, with optional input and output standardisation/scaling and the computation
    of output to input sensitivities.
    """

    def __init__(self, hidden_layer_size: int,
                 n_hidden_layers: int,
                 jacobian_regulariser: float = 0.0,
                 pytorch_init_seed: int = None):
        """
        :param hidden_layer_size: Number of neurons per hidden layer (could be extended to varying size)
        :param n_hidden_layers: Number of hidden layers
        :param jacobian_regulariser:
        :param pytorch_init_seed:
        """
        super(NeuralNetwork, self).__init__()

        if type(pytorch_init_seed) is int:
            torch.manual_seed(pytorch_init_seed)

        self.epochs_total = 0
        self.n_input_neurons = 4
        self.n_output_neurons = 1
        self.jacobian_regulariser = torch.tensor(jacobian_regulariser)
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size

        neurons_in_layers = [self.n_input_neurons] + [hidden_layer_size] * n_hidden_layers + [self.n_output_neurons]
        activation_function = 'ReLU'
        layer_dictionary = OrderedDict()

        layer_dictionary['input_standardisation'] = Standardise(self.n_input_neurons)

        for ii, (neurons_in, neurons_out) in enumerate(zip(neurons_in_layers[:-2], neurons_in_layers[1:-1])):
            layer_dictionary[f'dense_{ii}'] = nn.Linear(in_features=neurons_in,
                                                        out_features=neurons_out,
                                                        bias=True)

            if activation_function == "ReLU":
                layer_dictionary[f'activation_{ii}'] = nn.ReLU()
                nn.init.kaiming_normal_(layer_dictionary[f'dense_{ii}'].weight, mode='fan_in',
                                        nonlinearity='relu')

            elif activation_function == "Tanh":
                layer_dictionary[f'activation_{ii}'] = nn.Tanh()
                nn.init.kaiming_normal_(layer_dictionary[f'dense_{ii}'].weight, mode='fan_in',
                                        nonlinearity='tanh')
            else:
                raise Exception('Enter valid activation function! (ReLU or Tanh)')

        layer_dictionary['output_layer'] = nn.Linear(in_features=neurons_in_layers[-2],
                                                     out_features=neurons_in_layers[-1],
                                                     bias=True)
        nn.init.xavier_normal_(layer_dictionary['output_layer'].weight, gain=1.0)

        layer_dictionary['output_scaling'] = Scale(self.n_output_neurons)

        self.dense_layers = nn.Sequential(layer_dictionary)

    def standardise_input(self, input_statistics):
        self.dense_layers.input_standardisation.set_standardisation(mean=input_statistics[1],
                                                                    standard_deviation=input_statistics[0])

    def scale_output(self, output_statistics):
        self.dense_layers.output_scaling.set_scaling(mean=output_statistics[1],
                                                     standard_deviation=output_statistics[0])

    def forward(self, x):
        return self.dense_layers(x)

    def sensitivities(self, x):
        prediction, vjp = torch.autograd.functional.vjp(self.forward, x, v=torch.ones(x.shape[0], 1), create_graph=True)
        return prediction, vjp

from assignments.deeplearning.neural_topology import NeuralTopology

from assignments.deeplearning.neural_network import NeuralNetwork


class DigitNetwork:
  def __init__(self):
    input_dimensions = [28, 28]

    # topology = NeuralTopology.simple_topology(input_dimensions, [600, 400, 200], 10)
    convoluted_layers = [{'num_channels': 6, 'patch_size': 6, 'stride': 1},
                         {'num_channels': 12, 'patch_size': 5, 'stride': 2},
                         {'num_channels': 24, 'patch_size': 4, 'stride': 2}]
    topology = NeuralTopology.convoluted_topology(input_dimensions, convoluted_layers, [200], 10)

    self.neural_network = NeuralNetwork(topology)

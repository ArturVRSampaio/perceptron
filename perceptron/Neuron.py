from perceptron.Activation import step_activation


class Neuron:
    def __init__(self, bias: float, weights: [float], activation: callable = step_activation):
        self.bias = bias
        self.weights = weights
        self.activation = activation

    def __str__(self):
        return "bias: " + str(self.bias) + ". weights =" + str(self.weights)

    def predict(self, inputs: [float]) -> int:
        inputs_weights = zip(self.weights, inputs)
        weighted_sum = sum(weight * input for weight, input in inputs_weights) + self.bias
        return self.activation(weighted_sum)


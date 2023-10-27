import random

triple_bool_karnaugh = [[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1]]
triple_labels = [0, 1, 1, 1, 1, 1, 1, 1]

double_bool_karnaugh = [[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]]
double_labels = [0, 1, 1, 1]


def step_activation(value: float) -> int:
    if value >= 0:
        return 1
    else:
        return 0


class Neuron:
    def __init__(self, bias: float, weights: [float]):
        self.bias = bias
        self.weights = weights

    def __str__(self):
        return "bias: " + str(self.bias) + ". weights =" + str(self.weights)

    def predict(self, inputs: [float], activation_type: callable) -> int:
        inputs_weights = zip(self.weights, inputs)
        weighted_sum = sum(weight * input for weight, input in inputs_weights) + self.bias
        return activation_type(weighted_sum)


def is_neuron_ready(neuron: Neuron, input_array: [[float, float, float]], labels: [float],
                    verbose: bool = False) -> bool:
    error_flag = True
    for key, input in enumerate(input_array):
        label = labels[key]
        prediction = neuron.predict(input, step_activation)
        if label != prediction:
            error_flag = False
        if verbose:
            print(label, prediction)
    if verbose:
        print(error_flag)
    return error_flag


def train(neuron: Neuron, input_array: [[float]], labels: [float], learning_rate=0.1, epochs=500):
    for epoch in range(1, epochs + 1):
        for key, input in enumerate(input_array):
            label = labels[key]
            prediction = neuron.predict(input, step_activation)
            error = label - prediction

            for i in range(len(input)):
                neuron.weights[i] += learning_rate * error * input[i]
            neuron.bias += learning_rate * error

        if is_neuron_ready(neuron, input_array, labels):
            print("breaking on " + str(epoch) + " loop")
            break


double_neuron = Neuron(random.random(), [random.random(), random.random()])

train(double_neuron, double_bool_karnaugh, double_labels)

print("double neuron")
is_neuron_ready(double_neuron, double_bool_karnaugh, double_labels, True)

triple_neuron = Neuron(random.random(), [random.random(), random.random(), random.random()])

train(triple_neuron, triple_bool_karnaugh, triple_labels)

print("triple neuron")
is_neuron_ready(triple_neuron, triple_bool_karnaugh, triple_labels, True)

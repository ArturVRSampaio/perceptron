from perceptron.Neuron import Neuron


def is_neuron_ready(neuron: Neuron, input_array: [[float, float, float]], labels: [float],
                    verbose: bool = False) -> bool:
    error_flag = True
    for key, input in enumerate(input_array):
        label = labels[key]
        prediction = neuron.predict(input)
        if label != prediction:
            error_flag = False
        if verbose:
            print(input, label, prediction)
    if verbose:
        print(error_flag)
    return error_flag


def train(neuron: Neuron, input_array: [[float]], labels: [float], learning_rate=0.1, epochs=8000):
    for epoch in range(1, epochs + 1):
        for key, input in enumerate(input_array):
            label = labels[key]
            prediction = neuron.predict(input)
            error = label - prediction

            for i in range(len(input)):
                neuron.weights[i] += learning_rate * error * input[i]
            neuron.bias += learning_rate * error

        if is_neuron_ready(neuron, input_array, labels):
            print("breaking on " + str(epoch) + " loop")
            break

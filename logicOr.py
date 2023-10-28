import random

from perceptron.Neuron import Neuron
from perceptron.Tools import train, is_neuron_ready

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

double_neuron = Neuron(random.random(), [random.random(), random.random()])

train(double_neuron, double_bool_karnaugh, double_labels)

print("double neuron")
print(double_neuron)
is_neuron_ready(double_neuron, double_bool_karnaugh, double_labels, True)

triple_neuron = Neuron(random.random(), [random.random(), random.random(), random.random()])

train(triple_neuron, triple_bool_karnaugh, triple_labels)

print("triple neuron")
print(triple_neuron)
is_neuron_ready(triple_neuron, triple_bool_karnaugh, triple_labels, True)

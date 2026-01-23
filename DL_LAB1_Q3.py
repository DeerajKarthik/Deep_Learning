import numpy as np

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)    
        self.bias = bias

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return self.step_function(summation)

inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND gate perceptron
and_perceptron = Perceptron(weights=[1, 1], bias=-1.5)

print("AND Gate")
for x in inputs:
    print(f"{x} -> {and_perceptron.predict(x)}")

# OR gate perceptron
or_perceptron = Perceptron(weights=[1, 1], bias=-0.5)

print("\nOR Gate")
for x in inputs:
    print(f"{x} -> {or_perceptron.predict(x)}")


from engine import Node
import random

class Neuron:
    def __init__(self, numIn, activated):
        self.weight = [Node(random.uniform(-1,1)) for i in range(numIn)]
        self.bias = Node(random.uniform(-1,1))
        self.zeta = None
        self.activation = None
        self.activated = activated

    def parameters(self):
        return self.weight + [self.bias]
    
    def __call__(self, input):
        self.zeta = Node(0)
        for i in range(len(self.weight)):
            self.zeta += input[i]*self.weight[i]
        self.zeta += self.bias
        if self.activated: self.activation = (self.zeta).sigmoid()
        else: self.activation = (self.zeta).relu()
        return self.activation
    
class Layer:
    def __init__(self, numIn, numOut, activated):
        self.neurons = [Neuron(numIn, activated) for i in range(numOut)]

    def __call__(self, input):
        return [n(input) for n in self.neurons]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def reset_grad(self):
        for p in self.parameters():
            p.gradient = 0
    
    def loss(self, inputs, outputs):
        cleanedInputs = [[Node(elem) for elem in row] for row in inputs]
        cleanedOutputs = [Node(elem) for elem in outputs]
        predictions = [self(row) for row in cleanedInputs]
                
        data_sum=Node(0)
        for i in range(len(predictions)):
            a = cleanedOutputs[i]
            b = predictions[i][0]
            data_sum += cost(a, b)

        data_loss = data_sum * Node(1.0/len(predictions))
            
        return data_loss
    
    def gradient_descent(self, inputs, outputs, epochs, learn_rate):
        for k in range(epochs):
            total_loss = self.loss(inputs, outputs)
            print(f"step {k} loss {total_loss.data}")
            self.reset_grad()
            total_loss.autograd()
            for i in range(len(self.parameters())):
                val = self.parameters()[i]
                val.data -= learn_rate * val.gradient

def cost(a, b):
    return ((Node(1) - a*b).relu())
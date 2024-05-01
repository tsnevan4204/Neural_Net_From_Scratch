import math

class Node:
    def __init__(self, data, operation='', local_gradients=()):
        self.data = data
        self.local_gradients = local_gradients
        self.gradient = 0
        self.operation = operation
        self.children = []

    def __add__(self, other):
        sum = Node(self.data + other.data, '+', [(self, 1), (other, 1)])
        return sum
    
    def __mul__(self, other):
        product = Node(self.data * other.data, '*', [(self, other.data), (other, self.data)])
        return product
    
    def __sub__(self, other):
        difference = Node(self.data - other.data, '-', [(self, 1), (other, -1)])
        return difference
    
    def __repr__(self):
        return f"Node({self.data}, {self.gradient})"
    
    def relu(self):
        output = Node(max(self.data, 0), 'relu', [(self, 1 if self.data > 0 else 0)])
        return output
    
    def sigmoid(self):
        sig = lambda x: math.exp(x)/(1+math.exp(x))
        v = sig(self.data)
        output = Node(v, 'sigmoid', [(self, v*(1- v))])
        return output

    def autograd(self, parent_gradient = 1.0):
        for child, local_gradient in self.local_gradients:
            child.gradient += parent_gradient * local_gradient
            child.autograd(parent_gradient * local_gradient)
            self.children.append(child)
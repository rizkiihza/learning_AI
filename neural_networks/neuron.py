import value
import random

class Neuron:
    def __init__(self, nin: int):
        self.w = [value.Value(random.uniform(-1, 1), "w{i}".format(i=i)) for i in range(nin)]
        self.b = value.Value(random.uniform(-1, 1), "b")
    
    def __call__(self, x: list[value.Value]) -> value.Value:
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)        
        out = act.tanh() 

        return out
    
    def parameters(self) -> list[value.Value]:
        return self.w + [self.b]

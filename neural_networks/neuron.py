import value
import random
import visualizer

class Neuron:
    def __init__(self, nin: int):
        self.w = [value.Value(random.uniform(-1, 1), "w{i}".format(i=i)) for i in range(nin)]
        self.b = value.Value(random.uniform(-1, 1), "b")
    
    def __call__(self, x: list[value.Value]) -> value.Value:
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)        
        out = act.tanh() 

        return out

        
n = Neuron(3)
out = n([value.Value(0.2, "x0"), value.Value(-0.7, "x1"), value.Value(0, "x2")])
out.backpropagate()

visualizer.draw_directed(out)


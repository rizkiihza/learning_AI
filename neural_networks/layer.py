import neuron
import value

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [neuron.Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x: list[value.Value]) -> list[value.Value]:
        outs = [neuron(x) for neuron in self.neurons]
        return outs
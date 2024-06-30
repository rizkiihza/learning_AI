import layer
import value

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [layer.Layer(sizes[i-1], sizes[i]) for i in range(len(sizes[1:]))]
    
    def __call__(self, x: list[value.Value]) -> list[value.Value]:
        result = x
        for layer in self.layers:
            result = layer(result)
        
        return result
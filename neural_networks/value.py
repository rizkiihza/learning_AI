import math

class Value:
    def __init__(self, data, label, _parent=(), _op=""):
        self.data = data
        self.grad = 0
        self._parent= set(_parent)
        self._backward = lambda: None
        self._op = _op
        self.label = label
    
    def __repr__(self) -> str:
        return "Value=(data={data}, grad={grad})".format(data=self.data, grad=self.grad)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, str(other))
        out = Value(self.data + other.data, "{l1}+{l2}".format(l1=self.label, l2=other.label), (self, other), "+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only support float or int"
        out = Value(self.data ** other, "{l1}**{l2}".format(l1=self.label, l2=other), (self,), "**")

        def backward():
            other.grad = other * (self.data ** (other - 1)) * out.grad
        
        return out
        
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, str(other))
        out = Value(self.data * other.data, "{l1}{l2}".format(l1=self.label,l2=other.label), (self, other), "*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward

        return out

    def __rmul__(self, other):
        return self * other
    
    def tanh(self):
        tanh_data = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(tanh_data, "tanh({l1})".format(l1=self.label), (self,), "tanh")
        def backward():
            self.grad += (1 - tanh_data * tanh_data) * out.grad
        out._backward = backward

        return out

    def backpropagate(self):
        self.grad = 1
        nodes = [self]
        while len(nodes) > 0:
            front = nodes.pop()
            front._backward()

            for p in front._parent:
                nodes.append(p)

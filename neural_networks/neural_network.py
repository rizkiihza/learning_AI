import value
import mlp

class NeuralNetwork:
    def __init__(self, nin, nouts, h=0.0001):
        self.m = mlp.MLP(nin, nouts)
        self.h = h    
    def train(self, x_test: list[list[float]], y_target: list[float]):
        y_pred = [self.m(x) for x in x_test]
        loss = self.loss(y_pred, y_target)
        loss.backpropagate()

        print("loss: {loss:.2f}, y_pred={y_pred}".format(loss=loss.data, y_pred=[ypi.data for yp_list in y_pred for ypi in yp_list]))

        params = self.m.parameters()
        for p in params:
            p.data += -1 * self.h * p.grad
        
    def loss(self, y_pred: list[value.Value], y_target: list[float]) -> value.Value:
        return sum([(y_pred_i - y_target_i) ** 2 for y_pred_list, y_target_list in zip(y_pred, y_target) for \
            y_pred_i, y_target_i in zip(y_pred_list, y_target_list)])

nn = NeuralNetwork(3, [4, 3, 1])
x_train = [
    [0.3, -0.5, 0.7],
    [0.2, -0.2, 0.6],
    [0.4, -0.9, 0.5],
    [0.35, -0.4, 0.55]
] 
y_train = [
    [1], [-1], [1], [-1]
]

for i in range(100000):
    nn.train(x_train, y_train)


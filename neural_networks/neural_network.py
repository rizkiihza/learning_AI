import numpy as np
import matplotlib.pyplot as plt
import visualizer
import value

#x1 = value.Value(3, "x1")
#x2 = value.Value(9, "x2")
#w1 = value.Value(-0.2, "w1")
#w2 = value.Value(0.1, "w2")
#b = value.Value(0.5, "b")
#
#x1w1 = x1 * w1; x1w1.label = "x1w1"
#x2w2 = x2 * w2; x2w2.label = "x2w2"
#
#L = x1w1 + x2w2 + b; L.label = "L"
#tanh_L = L.tanh()
#tanh_L.backpropagate()
#visualizer.draw_directed(tanh_L)

x1 = value.Value(3, "x1")
x2 = x1 * x1
x2.backpropagate()
visualizer.draw_directed(x2)






import numpy as np
from nodes import NeuralNetwork

x=np.random.randn(200,784,)
y=np.random.randn(200)

forward=NeuralNetwork()
forward.init_params(10,6,784,2)
forward.forward_prop(x)

a=np.array([1,2,3])

print([a])
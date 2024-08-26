import numpy as np
from nodes import NeuralNetwork

x=np.random.randn(10,784)
y=np.random.randn(784)

forward=NeuralNetwork()
forward.init_params(10,10,784,10)
forward.forward_prop(x)
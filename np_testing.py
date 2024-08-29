import numpy as np
from nodes import NeuralNetwork

x=np.random.randn(4,2)
y=np.random.randn(4,2)

forward=NeuralNetwork()
forward.init_params(no_of_neurons=2,no_of_layers=10,no_of_input=2,no_of_output=3)
op=forward.forward_prop(x)
forward.back_prop(op,y,x)

a=np.array([1,2,3])

print([a])
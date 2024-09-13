import numpy as np
from nodes import NeuralNetwork

inp,op=3,3

x=np.random.randn(inp)
y=np.random.randn(op)

forward=NeuralNetwork()
forward.init_params(no_of_neurons=2,no_of_layers=10,no_of_input=inp,no_of_output=op)
op=forward.forward_prop(x)
forward.back_prop(op,x,y)

a=np.array([1,2,3])

print([a])

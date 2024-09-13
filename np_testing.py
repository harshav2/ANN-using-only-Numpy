import numpy as np
from nodes import NeuralNetwork

inp,op=3,3
no_vals=5

x=np.random.randn(no_vals,inp)
y=np.random.randn(no_vals,op)

forward=NeuralNetwork()
forward.init_params(no_of_neurons=2,no_of_layers=10,no_of_input=inp,no_of_output=op)
forward.fit(x,y)
import numpy as np
from nodes import NeuralNetwork

x=np.random.randn(200,784,)
y=np.random.randn(200,2)

forward=NeuralNetwork()
forward.init_params(10,1,784,2)
op=forward.forward_prop(x)
forward.back_prop(op,y,x)

a=np.array([1,2,3])

print([a])
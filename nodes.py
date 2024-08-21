import numpy as np
import os
from back_prop import BackProp
from forward_prop import ForwardProp

class NeuralNetwork(BackProp,ForwardProp):
    #network is to have 1 input variable x
    #network is to have 1 output variable z

    def __init__(self):
        self.input_weights=None
        self.hidden_weights=None
        self.output_weights=None
        self.hidden_bias=None
        self.output_bias=None
        self.intermediary_z=None

    def init_params(self,no_of_neurons,no_of_layers,no_of_input, no_of_output):
        #input layer parameters=1*neurons
        #intermediary layer paramaters=neurons*neurons (for "layers" number of layers)
        #output layer parameter=1*neurons
        self.input_weights=np.random.randn(no_of_input,no_of_neurons)
        self.hidden_weights=np.random.randn(no_of_layers-1,no_of_neurons,no_of_neurons)
        self.output_weights=np.random.randn(no_of_output,no_of_neurons)
        self.input_bias=np.random.randn(no_of_input)
        self.hidden_bias=np.random.randn(no_of_layers,no_of_neurons)
        self.output_bias=np.random.randn(no_of_output)
        
    def load_params(self):
        #loading stored weights from npy files
        self.input_weights = np.load(os.path.join('/data', "input_weights.npy"))
        self.hidden_weights = np.load(os.path.join('/data', "hidden_weights.npy"))
        self.output_weights = np.load(os.path.join('/data', "output_weights.npy"))
        self.input_bias= np.load(os.path.join('/data', "input_bias.npy"))
        self.hidden_bias= np.load(os.path.join('/data', "hidden_bias.npy"))
        self.output_bias= np.load(os.path.join('/data', "output_bias.npy"))


nn=NeuralNetwork()
try:
    nn.load_params()
except FileNotFoundError:
    nn.init_params(3, 3, 3, 3)

inp = np.array([1,1,1])

output = nn.forward_prop(inp)
nn.back_prop(output, target=[-2,-2,-2] , inp=inp)

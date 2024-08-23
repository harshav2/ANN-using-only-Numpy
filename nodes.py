import numpy as np
import os
from back_prop import BackProp
from forward_prop import ForwardProp
from train_and_predict import FitAndPredict

class NeuralNetwork(FitAndPredict):
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

    def save_params(self):
        np.save(os.path.join('/data', "input_weights.npy"), self.input_weights)
        np.save(os.path.join('/data', "hidden_weights.npy"), self.hidden_weights)
        np.save(os.path.join('/data', "output_weights.npy"), self.output_weights)
        np.save(os.path.join('/data', "input_bias.npy"), self.input_bias)
        np.save(os.path.join('/data', "hidden_bias.npy"), self.hidden_bias)
        np.save(os.path.join('/data', "output_bias.npy"), self.output_bias)


nn=NeuralNetwork()
try:
    nn.load_params()
    print(nn.output_bias)
except FileNotFoundError:
    nn.init_params(3, 3, 3, 3)

X = np.random.randn(50,3)
y = np.random.randn(50,3)
test=np.random.randn(10,3)

nn.fit(X, y)

final=nn.predict(test)
print(final)
import numpy as np
import os
from back_prop import BackProp
from forward_prop import ForwardProp
from train_and_predict import FitAndPredict

class NeuralNetwork(FitAndPredict):

    def __init__(self):
        self.input_weights=None
        self.hidden_weights=None
        self.output_weights=None
        self.hidden_bias=None
        self.output_bias=None
        self.intermediary_z=None

    def init_params(self,no_of_neurons,no_of_layers,no_of_input, no_of_output):
        self.input_weights=np.random.randn(no_of_input,no_of_neurons)
        self.hidden_weights=np.random.randn(no_of_layers,no_of_neurons,no_of_neurons)
        self.output_weights=np.random.randn(no_of_neurons,no_of_output)
        self.hidden_bias=np.random.randn(no_of_layers,no_of_neurons)
        self.output_bias=np.random.randn(no_of_output)
        
    def load_params(self):
        #loading stored weights from npy files
        self.input_weights = np.load(os.path.join('/data', "input_weights.npy"))
        self.hidden_weights = np.load(os.path.join('/data', "hidden_weights.npy"))
        self.output_weights = np.load(os.path.join('/data', "output_weights.npy"))
        self.hidden_bias= np.load(os.path.join('/data', "hidden_bias.npy"))
        self.output_bias= np.load(os.path.join('/data', "output_bias.npy"))

    def save_params(self):
        np.save(os.path.join('/data', "input_weights.npy"), self.input_weights)
        np.save(os.path.join('/data', "hidden_weights.npy"), self.hidden_weights)
        np.save(os.path.join('/data', "output_weights.npy"), self.output_weights)
        np.save(os.path.join('/data', "hidden_bias.npy"), self.hidden_bias)
        np.save(os.path.join('/data', "output_bias.npy"), self.output_bias)

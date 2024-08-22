import numpy as np
from forward_prop import sigmoid

class BackProp:
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def back_prop(self, output, target, inp, learning_rate=0.01):
        intermediary_a = sigmoid(self.intermediary_z)
        intermediary_aux = []

        #Calculating the auxilary variables
        final_error = output - target
        final_delta = final_error * self.sigmoid_derivative(output).T
        
        if len(final_delta)==1:
            current_aux = np.dot(self.output_weights.T, final_delta[0].T)
        else:
            current_aux = np.dot(self.output_weights.T, final_delta.T)
        intermediary_aux.append(current_aux)

        for i in range(len(self.hidden_weights) - 1, 0, -1):
            current_aux = current_aux * self.sigmoid_derivative(intermediary_a[i])
            intermediary_aux.append(current_aux)
            current_aux = np.dot(self.hidden_weights[i].T, current_aux)

        if len(final_delta)==1:
            self.output_weights -= learning_rate * np.dot(final_delta[0], intermediary_a[-1].T)
            self.output_bias -= learning_rate * final_delta[0]
        else:
            self.output_weights -= learning_rate * np.dot(final_delta, intermediary_a[-1].T)
            self.output_bias -= learning_rate * final_delta

        self.hidden_weights[i] -= learning_rate * np.dot(intermediary_aux[-2], intermediary_a[-1].T)
        self.hidden_bias[i] -= learning_rate * intermediary_aux[-2]

        for i in range(len(self.hidden_weights) - 1):
            self.hidden_weights[i] -= learning_rate * np.dot(intermediary_aux[-i-2], intermediary_a[-i-1].T)
            self.hidden_bias[i] -= learning_rate * intermediary_aux[-i-2]

        input_delta = intermediary_aux[-1] * self.sigmoid_derivative(intermediary_a[0])
        if len(inp)==1:
            self.input_weights -= learning_rate * np.dot(input_delta, inp[0])
            self.input_bias -= learning_rate * input_delta[0]
        else: 
            self.input_weights -= learning_rate * np.dot(input_delta, inp.T)
            self.input_bias -= learning_rate * input_delta
        
import numpy as np
from forward_prop import sigmoid

class BackProp:
    def sigmoid_derivative(self, x):
        x=np.array(x)
        return x * (1 - x)
    
    def back_prop(self, output, target, learning_rate=0.01):
        intermediary_a = sigmoid(self.intermediary_z[0])
        intermediary_aux = []

        #Calculating the auxilary variables
        final_error = output - target

        final_delta=np.multiply(self.sigmoid_derivative(output),final_error)
        print(final_delta,'\n',self.output_weights.T,'\n',intermediary_a[-1])
        current_aux=[]
        
        for i in range(len(self.output_weights.T)):
            current_aux.append(np.matmul(self.sigmoid_derivative(intermediary_a[-1]),np.dot(self.output_weights.T[i],final_delta)))
        print(current_aux)

























        # current_aux=[]
        # for i in range(len(self.output_weights.T)):
        #     current_aux.append(np.dot(self.output_weights.T[i],final_delta))
        # current_aux=np.array(current_aux)

        # intermediary_aux.append(current_aux)

        # for i in range(len(self.hidden_weights) - 1, -1, -1):
        #     current_aux = current_aux * self.sigmoid_derivative(intermediary_a[i])
        #     intermediary_aux.append(current_aux)
        #     current_aux = np.dot(self.hidden_weights[i].T, current_aux)

        # if len(final_delta)==1:
        #     self.output_weights -= learning_rate * np.dot(final_delta[0], intermediary_a[-1].T)
        #     self.output_bias -= learning_rate * final_delta[0]
        # else:
        #     self.output_weights -= learning_rate * np.dot(final_delta, intermediary_a[-1].T)
        #     self.output_bias -= learning_rate * final_delta

        # self.hidden_weights[i] -= learning_rate * np.dot(intermediary_aux[-2], intermediary_a[-1].T)
        # self.hidden_bias[i] -= learning_rate * intermediary_aux[-2]

        # for i in range(len(self.hidden_weights) - 1):
        #     self.hidden_weights[i] -= learning_rate * np.dot(intermediary_aux[-i-2], intermediary_a[-i-1].T)
        #     self.hidden_bias[i] -= learning_rate * intermediary_aux[-i-2]

        # input_delta = intermediary_aux[-1] * self.sigmoid_derivative(intermediary_a[0])
        # inp=self.intermediary_z[0]
        # if len()==1:
        #     self.input_weights -= learning_rate * np.dot(input_delta, inp[0])
        # else: 
        #     self.input_weights -= learning_rate * np.dot(input_delta, inp.T)
            
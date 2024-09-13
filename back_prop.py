import numpy as np
from forward_prop import sigmoid

class BackProp:
    def sigmoid_derivative(self, x):
        x=np.array(x)
        return x * (1 - x)
    
    def back_prop(self, input_, output, target, learning_rate=0.01):
        intermediary_a = sigmoid(self.intermediary_z)
        intermediary_aux = []

        #Calculating the auxilary variables
        final_error = output - target
        final_aux=np.multiply(self.sigmoid_derivative(output),final_error)
    
        current_aux=np.multiply(np.matmul(self.output_weights.T,final_aux),self.sigmoid_derivative(intermediary_a[-1]))
        intermediary_aux=[current_aux]

        for i in range(len(self.hidden_weights)):
            ind=-i-1
            current_aux=np.multiply(np.matmul(self.hidden_weights[i].T,current_aux),self.sigmoid_derivative(intermediary_a[ind-1]))
            intermediary_aux=[current_aux]+intermediary_aux

        input_aux=np.multiply(np.matmul(self.input_weights.T,current_aux),self.sigmoid_derivative(sigmoid(input_)))
        
        self.output_weights -= (learning_rate * np.outer(final_aux, intermediary_a[-1]))
        self.output_bias -= (learning_rate * final_aux)

        for i in range(len(self.hidden_weights)):
            self.hidden_weights[-i-1] -= (learning_rate * np.outer(intermediary_aux[-i-1], intermediary_a[-i-2]))
            self.hidden_bias[-i-1] -= (learning_rate * intermediary_aux[-i-1])
        
        self.input_weights -= (learning_rate * np.outer(intermediary_aux[0], input_))
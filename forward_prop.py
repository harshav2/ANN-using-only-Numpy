import numpy as np

def sigmoid(x):
    return 1/(1+(np.exp(-x)))

class ForwardProp:
    def forward_prop(self, inp_vector):
        self.intermediary_z=[]
        total_outputs=[]
        for inp in inp_vector:
            if len(inp)==1:
                z_inp=np.dot(self.input_weights, inp[0])
            else:
                z_inp=(np.dot(inp,self.input_weights))

            current_a=sigmoid(z_inp)
            int_z=[z_inp]

            for i in range(len(self.hidden_weights)):
                current_z=np.sum(np.dot(current_a,self.hidden_weights[i]))+self.hidden_bias[i]   

                int_z.append(current_z)
                current_a=sigmoid(current_z)  

            if len(self.output_weights[0])==1:
                output_z=np.sum(np.dot(self.output_weights.flatten(),current_a))+self.output_bias
            else:
                output_z=np.sum(np.dot(self.output_weights.T,current_a))+self.output_bias
            total_outputs.append(sigmoid(output_z))
            self.intermediary_z.append(int_z)
        self.intermediary_z=np.array(self.intermediary_z)
        return total_outputs

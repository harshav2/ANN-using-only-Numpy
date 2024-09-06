import numpy as np

def sigmoid(x):
    return 1/(1+(np.exp(-x)))

class ForwardProp:
    def forward_prop(self, inp):
        self.intermediary_z=[]

        z_inp=[]
        for i in range(len(self.input_weights)):
            z_inp.append(np.dot(self.input_weights[i],inp))
        z_inp=np.array(z_inp)+self.hidden_bias[0]

        current_a=sigmoid(z_inp)
        int_z=[z_inp]

        for i in range(len(self.hidden_weights)):
            weight=self.hidden_weights[i]
            bias=self.hidden_bias[i+1]

            current_z=[]
            for i in range(len(weight)):
                current_z.append(np.dot(weight[i],current_a))
            current_z=np.array(current_z)+bias

            current_a=sigmoid(current_z)  
            int_z.append(current_z)

        output_z=[]
        for i in range(len(self.output_weights)):
            output_z.append(np.dot(self.output_weights[i],current_a))
        output_z=np.array(output_z)+self.output_bias[0]
        self.intermediary_z.append(int_z)

        self.intermediary_z=np.array(self.intermediary_z)
        return sigmoid(output_z)

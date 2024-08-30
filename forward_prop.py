import numpy as np

def sigmoid(x):
    return 1/(1+(np.exp(-x)))

class ForwardProp:
    def forward_prop(self, inp_vector):
        self.intermediary_z=[]
        total_outputs=[]
        for inp in inp_vector:
            z_inp=[]
            for i in range(len(self.input_weights)):
                sum=0
                for j in range(len(inp)):
                    sum+=(self.input_weights[i][j]*inp[j])
                z_inp.append(sum)
            z_inp=np.array(z_inp)+self.hidden_bias[0]

            current_a=sigmoid(z_inp)
            int_z=[z_inp]

            for i in range(len(self.hidden_weights)):
                weight=self.hidden_weights[i]
                bias=self.hidden_bias[i+1]

                current_z=[]
                for i in range(len(weight)):
                    sum=0
                    for j in range(len(weight)):
                        sum+=(weight[i][j]*current_a[j])
                    current_z.append(sum)
                current_z=np.array(current_z)+bias

                current_a=sigmoid(current_z)  
                int_z.append(current_z)

            output_z=[]
            for i in range(len(self.output_weights)):
                sum=0
                for j in range(len(current_a)):
                    sum+=(self.output_weights[i][j]*current_a[j])
                output_z.append(sum)
            output_z=np.array(output_z)+self.output_bias[0]

            total_outputs.append(sigmoid(output_z))
            self.intermediary_z.append(int_z)
        self.intermediary_z=np.array(self.intermediary_z)
        return total_outputs

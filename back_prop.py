import numpy as np
from forward_prop import sigmoid

class BackProp:
    def back_prop(self,output,target,learning_rate=0.01):

        intermediary_a=sigmoid(self.intermediary_z)
        intermediary_aux=[]

        final_aux=output*(1-output)*(target-output)
        current_aux=intermediary_a[-1]*(1-intermediary_a[-1])*(self.output_weights*current_aux)
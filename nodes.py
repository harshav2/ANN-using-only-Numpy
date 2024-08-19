import numpy as np

class NodeLayer:
    def __init__(self,n):
        #n is number of neurons
        self.weights=np.random.randn(n,n)
        self.biases=np.random.randn(1,n)[0]
    
l=[5]+[20]*8 #u_xxx, u_xx, u_x, u, constant
neural_net=[NodeLayer() for _ in l]

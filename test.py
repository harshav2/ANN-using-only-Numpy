import numpy as np

x=np.random.randn(10,10)
y=np.random.randn(10)

print((x+y).shape)

from forward_prop import sigmoid
print(sigmoid(x+y).shape)
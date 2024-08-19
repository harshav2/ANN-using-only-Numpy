import numpy as np

def RELU(z):
    #returns relu of the input value/array
    #relu returns 0 for negative numbers, and the number itself otherwise
    return np.maximum(0,z)

import numpy as np

def RELU(x):
    return np.maximum(0, x)

def forward_prop(nn, x):
    z1 = np.dot(nn.input_weights, x)
    a1 = RELU(z1)

    current_a = a1
    intermediary_z = []
    intermediary_z.append(z1)

    for i in range(len(nn.hidden_weights)):
        weights = nn.hidden_weights[i]

        current_z = np.dot(weights, current_a)
        intermediary_z.append(current_z)

        current_a = RELU(current_z)
    
    for i, z in enumerate(intermediary_z):
        print(f"Layer {i+1} intermediary z:\n", z)
    
    output = np.dot(nn.output_weights, current_a)
    
    return output, intermediary_z
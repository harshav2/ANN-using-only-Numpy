import numpy as np

def sigmoid(x):
    return 1/(1+(np.exp(-x)))

def forward_prop(nn, x):
    print(nn.hidden_bias)
    z1 = np.dot(nn.input_weights, x)
    a1 = sigmoid(z1)

    current_a = a1
    intermediary_z = []
    intermediary_z.append(z1)

    for i in range(len(nn.hidden_weights)):
        weights = nn.hidden_weights[i]
        bias=nn.hidden_bias[i]

        current_z = np.dot(weights, current_a)+bias
        intermediary_z.append(current_z)

        current_a = sigmoid(current_z)
    
    for i, z in enumerate(intermediary_z):
        print(f"Layer {i+1} intermediary z:\n", z)
    
    output = sigmoid(np.dot(nn.output_weights, current_a) + nn.output_bias)
    print("Output:",output)
    
    return output, intermediary_z
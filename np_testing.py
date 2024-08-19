import os
import numpy as np

# Define the directory
output_dir = "/data"

# Load the numpy arrays
input_weights = np.load(os.path.join(output_dir, "input_weights.npy"))
hidden_weights = np.load(os.path.join(output_dir, "hidden_weights.npy"))
output_weights = np.load(os.path.join(output_dir, "output_weights.npy"))

# Print the arrays to verify
print("Input Weights:\n", input_weights)
print("\nHidden Weights:\n", hidden_weights)
print("\nOutput Weights:\n", output_weights)

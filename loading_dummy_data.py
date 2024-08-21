import os
import numpy as np

# Define the directory
output_dir = "/data"

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the numpy arrays
np.save(os.path.join(output_dir, "input_weights.npy"), np.random.randn(3))
np.save(os.path.join(output_dir, "hidden_weights.npy"), np.random.randn(2,3,3))
np.save(os.path.join(output_dir, "output_weights.npy"), np.random.randn(3))
np.save(os.path.join(output_dir, "input_bias.npy"), np.random.randn(1))
np.save(os.path.join(output_dir, "hidden_bias.npy"), np.random.randn(3,3))
np.save(os.path.join(output_dir, "output_bias.npy"), np.random.randn(1))

print("Files saved successfully.")

# np.save(os.path.join(output_dir, "input_weights.npy"), np.ones(3))
# np.save(os.path.join(output_dir, "hidden_weights.npy"), np.ones((2,3,3)))
# np.save(os.path.join(output_dir, "output_weights.npy"), np.ones(3))
# np.save(os.path.join(output_dir, "hidden_biases.npy"), np.ones((3,3)))
# np.save(os.path.join(output_dir, "output_bias.npy"), np.ones(1))
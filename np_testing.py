import numpy as np

# Example 2D array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Number to be added
number = 7

# Adding the number as a new row
new_row = np.array([[number]])
new_array = np.append(array, new_row, axis=0)

print(new_array)

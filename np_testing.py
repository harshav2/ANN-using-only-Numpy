import numpy as np

# Create an initial array
arr = np.array([1, 2, 3, 4])

# Append a single element
arr = np.append(arr, 5)
print("After appending a single element:", arr)

# Append multiple elements
arr = np.append(arr, [6, 7, 8])
print("After appending multiple elements:", arr)

# Append to a 2D array
arr_2d = np.array([[1, 2], [3, 4]])
new_row = np.array([[5, 6]])
arr_2d = np.append(arr_2d, new_row, axis=0)
print("After appending a row to 2D array:\n", arr_2d)

# Append a column to a 2D array
new_col = np.array([[7], [8], [9]])
arr_2d = np.append(arr_2d, new_col, axis=1)
print("After appending a column to 2D array:\n", arr_2d)
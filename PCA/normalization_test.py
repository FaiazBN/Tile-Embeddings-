import numpy as np

# Step 1: Load the array from file
arr = np.load("C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Kid Icarus_PCA_reduced.npy")
print("Original Array:")
print(arr)

# Step 2: Get max absolute values per column (axis=0)
max_abs_cols = np.max(np.abs(arr), axis=0)
print("\nMax Absolute:")
print(max_abs_cols)
# Step 3: Divide each column by its max absolute value
normalized_arr = arr / max_abs_cols  # broadcasts over columns
print("\nColumn-wise Normalized Array:")
print(normalized_arr[0])


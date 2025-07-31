import os
import numpy as np

# Load the data from the .npy file
data_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Original Data (Without Normalization)"
output_dir = os.path.join(data_dir, "Normalized")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith("_PCA_reduced.npy"):
        full_path = os.path.join(data_dir, filename)
        data = np.load(full_path)


        #assert data.shape[1] == 118, f"{filename} does not have 118 dimensions."

        # Calculate max absolute value per column
        max_abs_values = np.max(np.abs(data), axis=0)


        max_abs_values[max_abs_values == 0] = 1 # division by zero check

        # Normalize
        normalized_data = data / max_abs_values

        # Saving
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, normalized_data)

        #stats
        print(f"\n--- Stats for {filename} ---")
        print("Max absolute values per column:\n", max_abs_values)
        print("\nFirst row before normalization:\n", data[0])
        print("\nFirst row after normalization:\n", normalized_data[0])

        # Check min and max of entire normalized dataset
        global_min = np.min(normalized_data)
        global_max = np.max(normalized_data)
        print(f"\nValue range after normalization: [{global_min:.4f}, {global_max:.4f}]")


print(f"\nNormalization complete. Files saved in: {output_dir}")

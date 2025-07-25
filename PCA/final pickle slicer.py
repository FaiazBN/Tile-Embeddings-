import os
import pickle
import numpy as np

input_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/PCA_InverseTransformed"
output_root = os.path.join(os.path.dirname(input_dir), "Decoder Ready")

# Ensure output root directory exists
os.makedirs(output_root, exist_ok=True)

# Process each .pkl file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".pkl"):
        file_path = os.path.join(input_dir, filename)

        # Load the data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        n_rows = len(data)
        num_slices = n_rows // 9
        remainder = n_rows % 9

        print(f"Processing {filename}")
        print(f"  Total rows: {n_rows}")
        print(f"  Number of slices (9 each): {num_slices}")
        print(f"  Leftover rows (not saved): {remainder}")
        print()

        # Prepare output directory for this file
        basename = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_root, basename)
        os.makedirs(output_dir, exist_ok=True)

        # Save only complete slices
        for i in range(num_slices):
            slice_data = data[i * 9:(i + 1) * 9]
            slice_path = os.path.join(output_dir, f"Slice_{i + 1}.pkl")
            with open(slice_path, 'wb') as sf:
                pickle.dump(slice_data, sf)
import os
import numpy as np
import pickle

# === Paths ===
original_data_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Original Data (Without Normalization)"
vae_outputs_dir = os.path.join(os.path.dirname(original_data_dir), "Normalized", "..", "VAE Outputs")
output_denorm_dir = os.path.join(os.path.dirname(original_data_dir), "VAE Outputs De-normalized")

os.makedirs(output_denorm_dir, exist_ok=True)

# === Process ===
for filename in os.listdir(vae_outputs_dir):
    if filename.endswith("_reconstructed.pkl"):
        original_filename = filename.replace("_reconstructed.pkl", ".npy")
        base_name = filename.replace("_reconstructed.pkl", "")
        original_file_path = os.path.join(original_data_dir, original_filename)
        reconstructed_path = os.path.join(vae_outputs_dir, filename)

        if not os.path.exists(original_file_path):
            print(f"Skipping {filename} — original file not found: {original_filename}")
            continue

        print(f" De-normalizing: {filename}")

        # Load original (pre-normalized) data to get max_abs_values
        original_data = np.load(original_file_path)
        max_abs_values = np.max(np.abs(original_data), axis=0)
        max_abs_values[max_abs_values == 0] = 1  # Avoid division by zero

        # Print all 118 max abs values
        print("Max abs values used for de-normalization:")
        print(np.round(max_abs_values, 6))

        # Load reconstructed (normalized) data
        with open(reconstructed_path, "rb") as f:
            reconstructed = pickle.load(f)

        # De-normalize
        reconstructed_denorm = reconstructed * max_abs_values

        # Save de-normalized output
        output_path = os.path.join(output_denorm_dir, base_name + "_reconstructed_denormalized.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(reconstructed_denorm, f)


        print(f"Original shape       : {original_data.shape}")
        print(f"Reconstructed shape  : {reconstructed.shape}")
        print(f"Max abs values shape : {max_abs_values.shape}")
        print(f"De-normalized shape  : {reconstructed_denorm.shape}")
        print(f"Saved to            : {output_path}")
        print(f"Value range after de-normalization: [{np.min(reconstructed_denorm):.4f}, {np.max(reconstructed_denorm):.4f}]")



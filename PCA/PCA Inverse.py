import os
import pickle
import numpy as np

# === Paths ===
pca_model_path = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/fitted_pca_model.pkl"
input_folder = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/VAE Outputs De-normalized"
output_folder = input_folder.replace("VAE Outputs De-normalized", "PCA_InverseTransformed")

os.makedirs(output_folder, exist_ok=True)

# === Load PCA Model ===
with open(pca_model_path, "rb") as f:
    pca = pickle.load(f)
print("✅ PCA model loaded.")

# === Process Files ===
for filename in os.listdir(input_folder):
    if filename.endswith(".pkl") and "_PCA_reduced_reconstructed_denormalized" in filename:
        file_path = os.path.join(input_folder, filename)

        # Load the reduced data (n x 118)
        with open(file_path, "rb") as f:
            reduced_data = pickle.load(f)

        # Inverse transform to get original dimensions (n x 1024)
        restored_data = pca.inverse_transform(reduced_data)

        # Create output filename
        output_filename = filename.replace("_PCA_reduced_reconstructed_denormalized", "_PCA_inverse_transformed")
        output_path = os.path.join(output_folder, output_filename)

        # Save restored data
        with open(output_path, "wb") as f:
            pickle.dump(restored_data, f)

        print(f"Restored and saved: {output_filename}")

print("All files have been inverse transformed and saved.")

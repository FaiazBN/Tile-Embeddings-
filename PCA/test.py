import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

# === CONFIGURATION ===
game_folder = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Mega Man"
output_file = os.path.join(game_folder, "PCA_reduced_data.npy")
pca_components = 8

# === COLLECTION PHASE ===
all_data = []
level_count = 0
pickle_count = 0

print("Scanning game folder:", game_folder)

for level_name in os.listdir(game_folder):
    level_path = os.path.join(game_folder, level_name)
    if os.path.isdir(level_path):
        level_count += 1
        for file_name in os.listdir(level_path):
            if file_name.endswith(".pickle"):
                pickle_count += 1
                file_path = os.path.join(level_path, file_name)
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        data_np = np.array(data)
                        all_data.append(data_np)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

# Flatten to a single 2D array
if len(all_data) == 0:
    print("No valid pickle files found.")
    exit()

all_data_np = np.vstack(all_data)
print(f"\nTotal samples collected: {all_data_np.shape[0]}, feature dimension: {all_data_np.shape[1]}")

# === PCA REDUCTION PHASE ===
print("\nApplying PCA...")
pca = PCA(n_components=pca_components)
reduced_data = pca.fit_transform(all_data_np)

print("Reduced shape:", reduced_data.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# === SAVE OUTPUT ===
np.save(output_file, reduced_data)
print(f"\nSaved PCA-reduced data to: {output_file}")

# === SUMMARY ===
print(f"\nSummary:")
print(f"Levels processed: {level_count}")
print(f"Pickle files processed: {pickle_count}")
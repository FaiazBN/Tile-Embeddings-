import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

games_root = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data"
output_dir = os.path.join(games_root, "PCA_Output")
os.makedirs(output_dir, exist_ok=True)

all_data = []
game_index_map = {}
total_levels = 0
total_pickles = 0
total_samples = 0
current_index = 0

print("Starting data collection...")

for game_name in os.listdir(games_root):  # Check every game
    game_path = os.path.join(games_root, game_name)
    if not os.path.isdir(game_path) or game_name == "PCA_Output":
        continue

    game_sample_count = 0
    game_levels = 0
    game_pickles = 0

    for level_name in os.listdir(game_path):  # Check every level
        level_path = os.path.join(game_path, level_name)
        if not os.path.isdir(level_path):
            continue

        game_levels += 1
        for file_name in os.listdir(level_path):  # Check every pickle
            if file_name.endswith(".pickle"):
                file_path = os.path.join(level_path, file_name)
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)  # (9, 1024)
                        data_np = np.array(data)
                        all_data.append(data_np)
                        num_samples = data_np.shape[0]
                        game_sample_count += num_samples
                        game_pickles += 1
                except Exception as e:
                    print(f"  Error reading {file_path}: {e}")

    if game_sample_count > 0:
        game_index_map[game_name] = (current_index, current_index + game_sample_count)
        current_index += game_sample_count
        print(f"Collected from game '{game_name}': {game_sample_count} samples from {game_levels} levels and {game_pickles} pickle files.")

    total_levels += game_levels
    total_pickles += game_pickles
    total_samples += game_sample_count

print("\nFinished loading.")
print(f"Total games: {len(game_index_map)}")
print(f"Total levels: {total_levels}")
print(f"Total pickle files: {total_pickles}")
print(f"Total data samples: {total_samples}")


all_data_np = np.vstack(all_data)
print(all_data_np.shape)
print("\nRunning PCA on combined data...")
pca = PCA(n_components=250, svd_solver='randomized', random_state=42)
reduced_all_data = pca.fit_transform(all_data_np)
print(f"PCA reduced data to {reduced_all_data.shape[1]} dimensions from 1024.")
restored_data = pca.inverse_transform(reduced_all_data)

# Compare a few samples
print("\n--- Original vs Reconstructed (First 5 Samples) ---")
for i in range(5):
    print(f"\nSample {i + 1}")
    print("Original:", np.round(all_data_np[i][:10], 5))
    print("Restored:", np.round(restored_data[i][:10], 5))
    print(restored_data.shape)
    print("Variance:", np.round(all_data_np[i][:10] - restored_data[i][:10], 5))


# === Save Restored Data as 9-row Slices ===
restored_output_dir = "PCA_test_restored"
os.makedirs(restored_output_dir, exist_ok=True)

print("\nSaving restored slices...")

for game_name, (start_idx, end_idx) in game_index_map.items():
    game_data = restored_data[start_idx:end_idx]
    game_output_dir = os.path.join(restored_output_dir, game_name)
    os.makedirs(game_output_dir, exist_ok=True)

    num_slices = game_data.shape[0] // 9
    for i in range(num_slices):
        slice_data = game_data[i * 9 : (i + 1) * 9]
        slice_filename = os.path.join(game_output_dir, f"Slice_{i + 1}.pickle")
        with open(slice_filename, "wb") as f:
            pickle.dump(slice_data, f)

    leftover = game_data.shape[0] % 9
    print(f"{game_name}: {num_slices} slices saved, {leftover} leftover samples")

print("\nAll slices saved successfully!")

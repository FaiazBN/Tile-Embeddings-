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

for game_name in os.listdir(games_root): # Check every game
    game_path = os.path.join(games_root, game_name)
    if not os.path.isdir(game_path) or game_name == "PCA_Output":
        continue

    game_sample_count = 0
    game_levels = 0
    game_pickles = 0

    for level_name in os.listdir(game_path): # Check every level
        level_path = os.path.join(game_path, level_name)
        if not os.path.isdir(level_path):
            continue

        game_levels += 1
        for file_name in os.listdir(level_path): # Check every pickle
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

# Combine all data into one large array
all_data_np = np.vstack(all_data)  # (total_samples, 1024)

print("\nRunning PCA on combined data...")
pca = PCA(n_components=250, svd_solver='randomized', random_state=42)
reduced_all_data = pca.fit_transform(all_data_np)  # (total_samples, 8)

# Saving the model
# with open(os.path.join(output_dir, "fitted_pca_model.pkl"), "wb") as f:
#     pickle.dump(pca, f)

# # Save the big reduced array
# all_output_path = os.path.join(output_dir, "ALL_GAMES_PCA_reduced.npy")
# np.save(all_output_path, reduced_all_data)
# print(f"\nSaved combined PCA-reduced data to {all_output_path}")
print(f"Explained variance by each component: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
#
# Save individual game data
# print("\nSaving individual game data...")
# for game_name, (start, end) in game_index_map.items():
#     reduced_game_data = reduced_all_data[start:end]
#     game_output_path = os.path.join(output_dir, f"{game_name}_PCA_reduced.npy")
#     np.save(game_output_path, reduced_game_data)
#     print(f"  Saved '{game_name}' with shape {reduced_game_data.shape} to {game_output_path}")
#
# print("\n✅ Done.")


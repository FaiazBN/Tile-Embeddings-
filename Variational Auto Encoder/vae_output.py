import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sampling import Sampling

# === Paths ===
base_input_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Normalized"
base_output_dir = os.path.join(os.path.dirname(base_input_dir), "VAE Outputs")

os.makedirs(base_output_dir, exist_ok=True)

# === Load Models ===
encoder = load_model("vae_encoder.keras", compile=False, custom_objects={"Sampling": Sampling})
decoder = load_model("vae_decoder.keras", compile=False)

# === Process Files ===
for root, dirs, files in os.walk(base_input_dir):
    for file in files:
        if file.endswith(".npy"):
            input_file_path = os.path.join(root, file)
            print(f"\n🟡 Processing: {file}")

            data = np.load(input_file_path)  # shape: (N, 118)
            total_rows = data.shape[0]
            leftover = total_rows % 9

            # Pass entire file through VAE
            _, _, z = encoder(data)
            reconstructed = decoder(z).numpy()

            # Save output
            file_base_name = os.path.splitext(file)[0]
            output_path = os.path.join(base_output_dir, f"{file_base_name}_reconstructed.pkl")

            with open(output_path, "wb") as f:
                pickle.dump(reconstructed, f)

            # === Plot Every 1000th Row ===
            # for i in range(0, total_rows, 1000):
            #     plt.figure(figsize=(10, 4))
            #     plt.plot(data[i], label="Original", linewidth=2)
            #     plt.plot(reconstructed[i], label="Reconstructed", linestyle='dashed')
            #     plt.title(f"{file_base_name} - Sample {i}")
            #     plt.xlabel("Feature Index")
            #     plt.ylabel("Value")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()

            # === Summary ===
            print(f"✔ Total rows          : {total_rows}")
            print(f"✔ File shape          : {data.shape}")
            print(f"✔ Leftover rows (9%)  : {leftover}")
            print(f"📁 Saved to           : {output_path}")
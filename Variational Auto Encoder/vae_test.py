import numpy as np
import matplotlib.pyplot as plt
from sampling import Sampling
from keras.models import load_model


data_path = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Normalized/Super Mario Kart_PCA_reduced.npy"
data = np.load(data_path)

sample = np.expand_dims(data[0], axis=0)

encoder = load_model("vae_encoder.keras", compile=False, custom_objects={"Sampling": Sampling})
decoder = load_model("vae_decoder.keras", compile=False)

z_mean, z_log_var, z = encoder(sample)
reconstructed = decoder(z)

# Original vs Reconstructed
original_vector = sample[0]
reconstructed_vector = reconstructed.numpy()[0]

# Print first 10 values
print("Original:     ", np.round(original_vector[:117], 4))
print("Reconstruction:", np.round(reconstructed_vector[:117], 4))

# === Step 5: Plot ===
plt.figure(figsize=(10, 4))
plt.plot(original_vector, label="Original", linewidth=2)
plt.plot(reconstructed_vector, label="Reconstructed", linestyle='dashed')
plt.title("Original vs Reconstructed (Sample 0)")
plt.xlabel("Feature Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
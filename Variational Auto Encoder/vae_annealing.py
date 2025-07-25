import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, ops, models, optimizers, losses
from keras.src.saving import load_model
from keras.saving import register_keras_serializable
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.callbacks import ModelCheckpoint

os.environ["KERAS_BACKEND"] = "tensorflow"


@register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# == VAE Model ==
class VAE(models.Model):
    def __init__(self, encoder, decoder, kl_start=0.0, kl_max=1.0, kl_anneal_epochs=20, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # KL annealing
        self.kl_weight = kl_start
        self.kl_max = kl_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.current_epoch = 0

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def update_kl_weight(self):
        if self.current_epoch < self.kl_anneal_epochs:
            self.kl_weight = (self.current_epoch / self.kl_anneal_epochs) * self.kl_max
        else:
            self.kl_weight = self.kl_max
        self.current_epoch += 1

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            mse_loss_fn = keras.losses.MeanSquaredError(
                reduction="sum_over_batch_size", name="mean_squared_error"
            )
            reconstruction_loss = mse_loss_fn(data, reconstruction)

            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)

        mse_loss_fn = keras.losses.MeanSquaredError(
            reduction="sum_over_batch_size", name="mean_squared_error"
        )
        reconstruction_loss = mse_loss_fn(data, reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        kl_loss = ops.mean(ops.sum(kl_loss, axis=1))

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "kl_weight": self.kl_weight
        }



def build_encoder(input_dim=250, latent_dim=16):
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="tanh")(encoder_inputs)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.Dense(64, activation="tanh")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    #encoder.summary()
    return encoder


def build_decoder(output_dim=250, latent_dim=16):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation="tanh")(latent_inputs)
    x = layers.Dense(128, activation="tanh")(x)
    x = layers.Dense(256, activation="tanh")(x)
    decoder_outputs = layers.Dense(output_dim, activation="tanh")(x)
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
    #decoder.summary()
    return decoder


def load_pca_data_from_files(data_dir="."):
    train_data = []
    test_data = []
    current_index = 0

    for filename in os.listdir(data_dir):
        if filename.endswith("_PCA_reduced.npy"):
            print("Processing " + filename)
            full_path = os.path.join(data_dir, filename)
            data = np.load(full_path)
            num_samples = data.shape[0]
            train_end = int(num_samples * 0.9)  # Getting the first 90% of the data for each game
            print(train_end)
            current_index += train_end
            #print(current_index)
            train_data.append(data[:train_end])
            test_data.append(data[train_end:])

    x_train = np.concatenate(train_data, axis=0).astype("float32")
    x_test = np.concatenate(test_data, axis=0).astype("float32")
    #print(x_train)
    #sample_index = 274977
    #x_train = x_train[sample_index: sample_index + 1]
    #x_test = x_test[sample_index: sample_index + 1]

    print(x_train)
    #print(x_test)

    return x_train, x_test


def train_vae(encoder, decoder, x_train, x_test, learning_rate=0.0001, total_epochs=100):
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    vae = VAE(encoder, decoder, kl_start=0.0, kl_max=1.0, kl_anneal_epochs=20)
    vae.compile(optimizer=optimizer)

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        vae.update_kl_weight()  # Gradually increase KL weight
        vae.fit(x_train, epochs=1, batch_size=1, validation_data=(x_test, x_test))

    encoder.save("vae_encoder.keras")
    decoder.save("vae_decoder.keras")

    return vae


def load_vae():
    encoder = load_model("vae_encoder.keras", custom_objects={"Sampling": Sampling})
    decoder = load_model("vae_decoder.keras")
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=optimizers.Adam(), loss=None)  # Changed
    return vae


def plot_latent_space(vae, n=30, figsize=10):
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample, verbose=0)
            print(f"Sample [{xi:.2f}, {yi:.2f}]: {x_decoded[0][:5]}...")


# == Main ==
def main():
    if os.path.exists("vae_encoder.keras") and os.path.exists("vae_decoder.keras"):
        print("Loading trained model...")
        vae = load_vae()
    else:
        print("Loading custom PCA-reduced game data...")
        data_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Normalized"
        x_train, x_test = load_pca_data_from_files(data_dir)

        print("Training VAE on custom data...")
        print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
        encoder = build_encoder()
        decoder = build_decoder()
        print(device_lib.list_local_devices())
        print(tf.__version__)
        #vae = train_vae(encoder, decoder, x_train, x_test)

    #plot_latent_space(vae)


if __name__ == "__main__":
    main()

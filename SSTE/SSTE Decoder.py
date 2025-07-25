# importing libraries
import os
import sys
import json
import glob
import pickle
import re
from collections import Counter
#import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from IPython.display import display
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, optimizers, backend as K
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    Dropout,
    Input,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    Embedding,
    LSTM,
    Add,
)
from tensorflow.keras.preprocessing import sequence, image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model, model_to_dot, to_categorical

#from sklearn.model_selection import train_test_split

# from annoy import AnnoyIndex


sste_latent_dim = 1024

image_shape_before_f = None
text_shape_before_c = None
num_clusters = 10

debug = 1
def get_encoder():
    global image_shape_before_f, text_shape_before_c
    image_encoder_input = keras.Input(shape=(48, 48, 3), name="image_input")
    image_encoder_conv_layer1 = layers.Conv2D(32, strides=3, kernel_size=(3,3), name="iencode_conv1")(image_encoder_input)
    image_encoder_norm_layer1 = layers.BatchNormalization()(image_encoder_conv_layer1)
    image_encoder_actv_layer1 = layers.ReLU()(image_encoder_norm_layer1)
    image_encoder_conv_layer2 = layers.Conv2D(32, (3,3), padding="same", name="iencoder_conv2")(image_encoder_actv_layer1)
    image_encoder_norm_layer2 = layers.BatchNormalization()(image_encoder_conv_layer2)
    image_encoder_actv_layer2 = layers.ReLU()(image_encoder_norm_layer2)
    image_encoder_conv_layer3 = layers.Conv2D(16, (3,3), padding="same", name="iencoder_conv3")(image_encoder_actv_layer2)
    image_encoder_norm_layer3 = layers.BatchNormalization()(image_encoder_conv_layer3)
    image_encoder_actv_layer3 = layers.ReLU()(image_encoder_norm_layer3)

    image_shape_before_flatten = tf.keras.backend.int_shape(image_encoder_actv_layer3)[1:]
    image_shape_before_f = tf.keras.backend.int_shape(image_encoder_actv_layer3)[1:]
    image_flatten = layers.Flatten(name="image_flatten_layer")(image_encoder_actv_layer3)

    # text encoder
    text_encoder_input = layers.Input(shape=(8,))
    text_encoder_dense_layer1 = layers.Dense(32, activation="tanh", name="tencode_dense1")(text_encoder_input)
    text_encoder_dense_layer2 = layers.Dense(16, activation="tanh", name="tencode_dense2")(text_encoder_dense_layer1)

    text_shape_before_concat = tf.keras.backend.int_shape(text_encoder_dense_layer2)[1:]
    text_shape_before_c = tf.keras.backend.int_shape(text_encoder_dense_layer2)[1:]

    # image-text concatenation
    image_text_concat = layers.Concatenate(name="image_text_concatenation")([image_flatten, text_encoder_dense_layer2])
    image_text_concat = layers.Dense(sste_latent_dim, activation="tanh", name="embedding_dense_1")(image_text_concat)

    z_mean = layers.Dense(sste_latent_dim, name="z_mean")(image_text_concat)
    #z_log_var = layers.Dense(sste_latent_dim, name="z_log_var")(image_text_concat)
    #z = Sampling()([z_mean, z_log_var])
    #encoder = keras.Model([image_encoder_input, text_encoder_input], [z_mean, z_log_var, z], name="encoder")
    encoder = keras.Model([image_encoder_input, text_encoder_input], z_mean, name="encoder")

    if debug == 1:
        print(encoder.summary())
    return encoder



def get_decoder():
    global image_shape_before_f, text_shape_before_c
    latent_input = keras.Input(shape=(sste_latent_dim,))

    # decoder for image
    #assert image_shape_before_f != None
    image_shape_before_flatten = image_shape_before_f
    image_y = layers.Dense(units=np.prod(image_shape_before_flatten), name="image_dense")(latent_input)
    image_y = layers.Reshape(target_shape=image_shape_before_flatten, name="image_reshape")(image_y)
    image_decoder_convt_layer1 = layers.Conv2DTranspose(16, (3,3), padding="same", name="idecode_conv1")(image_y)
    image_decoder_norm_layer1 = layers.BatchNormalization(name="idecode_norm1")(image_decoder_convt_layer1)
    image_decoder_actv_layer1 = layers.ReLU(name="idecode_relu1")(image_decoder_norm_layer1)
    image_decoder_convt_layer2 = layers.Conv2DTranspose(32, (3,3), padding="same", name="idecode_conv2")(image_decoder_actv_layer1)
    image_decoder_norm_layer2 = layers.BatchNormalization(name="idecode_norm2")(image_decoder_convt_layer2)
    image_decoder_actv_layer2 = layers.ReLU(name="idecode_relu2")(image_decoder_norm_layer2) 
    image_decoder_output = layers.Conv2DTranspose(3, (3,3), padding="same", name="image_output_layer")(image_decoder_actv_layer2)

    # decoder for text
    text_decoder_dense_layer1 = layers.Dense(16, activation="tanh", name="tdecode_dense1")(latent_input)
    assert text_shape_before_c != None
    text_shape_before_concat = text_shape_before_c
    text_reshape = layers.Reshape(target_shape=text_shape_before_concat, name="text_reshape")(text_decoder_dense_layer1)
    text_decoder_dense_layer2 = layers.Dense(32, activation="tanh", name="tdecode_dense2")(text_reshape)
    text_decoder_output = layers.Dense(8, activation="sigmoid", name="text_output_layer")(text_decoder_dense_layer2)

    image_output_flatten = layers.Flatten(name="image_output_flatten")(image_decoder_output)
    d_image_text_concat = layers.Concatenate(name="image_text_output")([image_output_flatten, text_decoder_output])

    predicted_cluster_assignments = layers.Dense(num_clusters, activation="softmax", name="cluster_output_layer")(d_image_text_concat)

    decoder = keras.Model(latent_input, [image_decoder_output, text_decoder_output, predicted_cluster_assignments])
    if debug == 1:
        print(decoder.summary())
    return decoder

encoding_model = get_encoder()
encoding_model.load_weights("encoder_weights_vSSTE_2.0_rs13_epoch_50.weights.h5", skip_mismatch= True)
print("Loaded Encoder Model from the Disk")

decoding_model = get_decoder()
decoding_model.load_weights("decoder_weights_vSSTE_2.0_rs13_epoch_50.weights.h5", skip_mismatch=True)
print("Loaded Decoder Model from the Disk")




def decode_latents_from_pickle(pickle_path, output_dir):
    # Load pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    data= np.array(data)
    print("Loaded latent vectors of shape:", data.shape)

    decoded_images, decoded_texts, cluster_probs = decoding_model.predict(data)
    print("Decoded images:", decoded_images.shape)

    os.makedirs(output_dir, exist_ok=True)

    tiles = []
    for i, full_img in enumerate(decoded_images):
        rgb_img = array_to_img(full_img)

        #plt.imshow(rgb_img)
        #plt.show()

        # Convert to HSV and back using cv2
        hsv_array = np.array(rgb_img)
        #hsv_img = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        rgb_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2RGB)
        rgb_img = Image.fromarray(rgb_array)
        #plt.imshow(rgb_img)
        #plt.show()
        tiles.append(rgb_img)
        #print(tiles)

    tile_w, tile_h = tiles[0].size
    #print(tile_w, tile_h)
    combined_img = Image.new("RGB", (3 * tile_w, 3 * tile_h)) # Blank Image

    for idx, tile in enumerate(tiles):
        row = idx // 3
        col = idx % 3
        combined_img.paste(tile, (col * tile_w, row * tile_h)) # Tile image
    combined_img = combined_img
    plt.imshow(combined_img)
    plt.show()
    combined_path = os.path.join(output_dir, "combined_3x3_tiles.png")
    combined_img.save(combined_path)
    print(f"Saved combined image: {combined_path}")
    print(decoded_texts)
    # Save decoded outputs
    np.save(os.path.join(output_dir, "decoded_texts.npy"), decoded_texts)
    #np.save(os.path.join(output_dir, "cluster_probs.npy"), cluster_probs)
    print("Saved decoded text and cluster predictions.")

pickle_path = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Decoder Ready/Super Mario Kart_PCA_inverse_transformed/Slice_1.pkl"
pickle_path_org = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Kart/Level 1/Slice_11.pickle"
pickle_pca_test = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/PCA/PCA_test_restored/Super Mario Kart/Slice_1.pickle"
output_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/SSTE"

new_smk_pickle_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/Result Calculation/smk_new.pickle"
decode_latents_from_pickle(new_smk_pickle_dir, output_dir)
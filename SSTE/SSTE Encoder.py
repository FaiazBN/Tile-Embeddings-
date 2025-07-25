#importing libraries
import os
import sys
import json
import glob
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

from PIL import Image, ImageOps

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

from sklearn.model_selection import train_test_split

#from annoy import AnnoyIndex



# Loading the Encoder Model
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



encoding_model = get_encoder()
encoding_model.load_weights("encoder_weights_vSSTE_2.0_rs13_epoch_50.weights.h5", skip_mismatch= True)
print("Loaded Encoder Model from the Disk")



# Run the Model and genarations of pickle


def get_image(path):
    img_without_border = load_img(path)
    img = Image.open(path)
    img_with_border = ImageOps.expand(img_without_border, border=16, fill="black")
    return img_without_border, img_with_border

def level_image_unroll(level_array_padded):
    level_image_unrolled = []
    image_h, image_w, image_c = level_array_padded.shape
    for x in range(0, image_w - 32, 16):
        for y in range(0, image_h - 32, 16):
            context_tile = level_array_padded[y : y + 48, x : x + 48, :]
            level_image_unrolled.append(context_tile)
    return np.array(level_image_unrolled)

def build_game_dataframe(current_game, game_image_dir, image_extension):
    # Get all image paths directly
    image_paths = glob.glob(os.path.join(game_image_dir, "*" + image_extension))

    # Extract image IDs cleanly from these paths
    image_ids = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

    # Build the dataframe using the already correct paths
    game_data = pd.DataFrame(columns=["image_path"])
    game_data["image_path"] = image_paths

    assert game_data.shape[0] == len(image_ids)
    print("\nAll Levels Loaded")
    print("\nTotal Levels for game ", current_game, " detected are ", len(image_ids))
    return game_data, image_ids

def generate_unified_rep(current_game, loaded_game_data, game_image_dir, save_dir, affordance_dir):
    ptr = 0
    idx2embed_map = {}
    idx2tile_map = {}
    mapped_text = np.load(affordance_dir)

    for idx in range(len(loaded_game_data)):
        image_path = loaded_game_data.loc[idx]["image_path"]
        level_id = image_path.split("/")[-1].split(".")[0]
        print("\nProcessing level", level_id)

        # Extract the clean level number
        try:
            level_number = int(level_id[6:].split('\\')[0]) - 1
        except ValueError:
            print(f"Invalid level id format: {level_id}")
            continue

        # Use only the slice name for saving file
        slice_id = level_id.split('\\')[-1]

        level_img, level_img_padded = get_image(image_path)
        level_img = level_img.convert("HSV")
        level_img_padded = level_img_padded.convert("HSV")

        level_array = img_to_array(level_img)
        level_array_padded = img_to_array(level_img_padded)

        if level_array.shape[0] % 16 != 0 or level_array.shape[1] % 16 != 0:
            print("Tile Pixel size is too small, Skipping Level ", level_id)
            continue

        #assert level_array.shape[0] % 16 == 0
        #assert level_array.shape[1] % 16 == 0

        level_h = level_array.shape[0] / 16
        level_w = level_array.shape[1] / 16
        print("Height ", level_h, "Width ", level_w)

        level_image_expanded = level_image_unroll(level_array_padded)
        print("Expanded level images ", level_image_expanded.shape)

        if level_h < 3.0 or level_w < 3.0:
            print("Level ", level_id, " is too small")
            continue

        #print(mapped_text[level_number][4])
        encoded_level = encoding_model.predict([level_image_expanded, mapped_text[level_number]])

        for i in range(len(encoded_level)):
            tile_embedding = encoded_level[i]
            tile_sprite = level_image_expanded[i].reshape(48, 48, 3)[16:32, 16:32, :]
            idx2embed_map[ptr] = tile_embedding
            idx2tile_map[ptr] = tile_sprite
            ptr += 1

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"{slice_id}.pickle")
        with open(file_path, "wb") as handle:
            pickle.dump(encoded_level, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved ", slice_id, " successfully!")

    # Save mappings in a 'mappings' subfolder
    mappings_dir = os.path.join(save_dir, "mappings")
    os.makedirs(mappings_dir, exist_ok=True)

    embed_map_path = os.path.join(mappings_dir, "idx2embed.pickle")
    with open(embed_map_path, "wb") as handle:
        pickle.dump(idx2embed_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Index to Embedding map saved successfully!")

    tile_map_path = os.path.join(mappings_dir, "idx2tile.pickle")
    with open(tile_map_path, "wb") as handle:
        pickle.dump(idx2tile_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Index to Tile map saved successfully!")

    print("Extracted unified representation for game ", current_game)


if __name__ == "__main__":
    current_game = "Super Mario Bros"
    game_image_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/Sliced Levels/Super Mario Bros 2 (Japan)/Level 15"
    affordance_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/Numpy Data/Super Mario Bros 2 (Japan)/Level 15/affordance_data.npy"
    save_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Bros 2 (Japan)/Level 15"
    loaded_game_data, identifiers = build_game_dataframe(
        current_game,
        game_image_dir,
        ".png")

    generate_unified_rep(current_game,loaded_game_data,game_image_dir, save_dir, affordance_dir)

    print("Saved Super Mario Kart Unified Representation!")
import json

from PIL import Image, ImageOps
import numpy as np
import os

# Slices up the image based on the size given
def slice_image(image_path, output_folder, tile_size=48):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Determine output directory
    # output_folder = os.path.dirname(image_path + "\\Level 1")

    # Loop through the grid and save each slice
    count = 1
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            right = min(col + tile_size, width)
            lower = min(row + tile_size, height)

            # Crop the image
            cropped_image = image.crop((col, row, right, lower))

            # Save the slice
            slice_path = os.path.join(output_folder, f"Slice_{count}.png")
            cropped_image.save(slice_path)
            print(f"Saved: {slice_path}")
            count += 1


# Upscales the image using a linear scaling factor
def upscale_image(image_path, scale_factor=2, resample=Image.NEAREST):
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    upscaled_image = image.resize(new_size, resample=resample)

    output_path = os.path.splitext(image_path)[0] + f"_upscaled.png"
    upscaled_image.save(output_path)
    print(f"Upscaled image saved to {output_path}")
    return output_path


# def convert_to_hsv(image_path):
#     image = Image.open(image_path).convert("HSV")
#
#     output_path = os.path.splitext(image_path)[0] + "_hsv.png"
#     image.save(output_path)
#     print(f"HSV image saved to {output_path}")
#     return output_path


# Reads the JSON file given  for the game and loads up the affordance data
def load_tile_affordances(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    affordance_mapping = {
        "breakable": 0, "climbable": 1, "collectable": 2, "hazard": 3,
        "moving": 4, "passable": 5, "portal": 6, "solid": 7
    }

    tile_vectors = {}
    for tile, affordances in data["tiles"].items():
        vector = np.zeros(8)
        for affordance in affordances:
            if affordance in affordance_mapping:
                vector[affordance_mapping[affordance]] = 1
        tile_vectors[tile] = vector

    return tile_vectors


# Returns np array with all the affordance data. The 8 length vector for all the tiles
def affordance_text_level(text_file_path, json_path, output_dir, affordance_file_name, tile_size=3):
    tile_vectors = load_tile_affordances(json_path)
    #print(tile_vectors)

    with open(text_file_path, "r") as file:
        lines = file.read().strip().split("\n")

    height = len(lines)
    width = len(lines[0]) if height > 0 else 0


    slices = {}
    count = 1

    affordance_data = []

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            slice_vectors = []
            for i in range(tile_size):
                for j in range(tile_size):
                    if row + i < height and col + j < width:
                        tile = lines[row + i][col + j]
                        slice_vectors.append(tile_vectors.get(tile, np.zeros(8)))
                    else:
                        slice_vectors.append(np.zeros(8))  # Fill empty space with zero vectors

            slice_array = np.array(slice_vectors).reshape(9, 8)
            slices[count] = slice_array
            affordance_data.append(slice_array)
            count += 1

    # Save affordance data as a .npy file if it doesn't exist
    affordance_file_path = os.path.join(output_dir, affordance_file_name)
    if not os.path.exists(affordance_file_path):
        np.save(affordance_file_path, np.array(affordance_data))
        print(f"Saved affordance data to {affordance_file_path}")
    else:
        print(f"Affordance data already exists at {affordance_file_path}")

    return slices, affordance_file_path

# Chops the text based representation of the level into given size (Eg.: 3x3)
def chop_text_level(file_path, output_path, tile_size=3):
    with open(file_path, "r") as file:
        lines = file.read().strip().split("\n")

    height = len(lines)
    width = len(lines[0]) if height > 0 else 0

    chopped_sections = []

    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            section = []
            for i in range(tile_size):
                if row + i < height:
                    section.append(lines[row + i][col:col + tile_size])
                else:
                    section.append("")  # Empty row if out of bounds

            chopped_sections.append(section)

    np_array = np.array(chopped_sections, dtype=object)
    np.save(output_path, np_array)

    print(f"Saved chopped text level as NumPy array at {output_path}")

    return np_array

    #return chopped_sections



image_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk1.png"
upscaled_img_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk1_upscaled.png"
hsv_img_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk1_upscaled_hsv.png"

slice_output_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\Level 1"

text_file_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk1.txt"
json_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk.json"
json_path_test_01 = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smk_test01.json"
affordance_npy_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer"

#upscale_image(image_path)
#convert_to_hsv(upscaled_img_path)
#slice_image(upscaled_img_path, slice_output_path)
text_slices_saved_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer"
affordance_file_name = "affordance_data_test_01.npy"
#text_slices = chop_text_level(text_file_path, text_slices_saved_path)
#print(len(text_slices))

#print(slices[395])
#print(slices)
#print()
#print()
#mapped_text = np.load(affordance_file)
#print("Loaded affordance data:", mapped_text)
#print("Length of affordance data:", len(mapped_text))

# Mario Game
# Upscale
# mario_image_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\super_mario_land_11.png"
# upscale_image(mario_image_path)


# Slicing
# mario_slice_output_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\Mario Level"
# mario_upscaled_img_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\super_mario_land_11_upscaled.png"
# slice_image(mario_upscaled_img_path, mario_slice_output_path)


# Getting the affordance
#mario_text_file_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\super_mario_land_11.png.txt"
#mario_json_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\smb.json"
#mario_affordance_file_name = "mario_affordance_data.npy"
#affordance_text_level(mario_text_file_path, mario_json_path, affordance_npy_path, mario_affordance_file_name)

# Getting the 3x3s
mario_text_file_path = "C:\\Users\\faiaz\\OneDrive\\Desktop\\Research Stuffs\\FAIAZ Tile Embeddings Stuff\\Python Image Slicer\\super_mario_land_11.png.txt"
chop_text_level(mario_text_file_path, text_slices_saved_path)
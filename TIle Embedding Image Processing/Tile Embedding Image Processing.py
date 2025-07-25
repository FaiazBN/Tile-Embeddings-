import os
import json
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def upscale_image(image_path, output_dir, scale_factor=2, resample=Image.NEAREST):
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    upscaled_image = image.resize(new_size, resample=resample)
    output_path = os.path.join(output_dir, os.path.basename(image_path).rsplit(".", 1)[0] + "_upscaled.png")
    upscaled_image.save(output_path)
    print(f"Upscaled image saved to {output_path}")
    return output_path

def downscale_image(image_path, output_dir, scale_factor=0.5, resample=Image.LANCZOS):
    image = Image.open(image_path)
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    downscaled_image = image.resize(new_size, resample=resample)
    output_path = os.path.join(output_dir, os.path.basename(image_path).rsplit(".", 1)[0] + "_downscaled.png")
    downscaled_image.save(output_path)
    print(f"Downscaled image saved to {output_path}")
    return output_path

def slice_image(image_path, output_folder, tile_size=48):
    image = Image.open(image_path)
    width, height = image.size
    count = 1
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            right = min(col + tile_size, width)
            lower = min(row + tile_size, height)
            cropped_image = image.crop((col, row, right, lower))
            slice_path = os.path.join(output_folder, f"Slice_{count}.png")
            cropped_image.save(slice_path)
            count += 1
    print(f"Sliced image saved in {output_folder}")

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


def affordance_text_level(text_file_path, json_path, output_dir, output_name="affordance_data.npy", tile_size=3):
    tile_vectors = load_tile_affordances(json_path)
    with open(text_file_path, "r") as file:
        lines = file.read().strip().split("\n")

    height = len(lines)
    width = len(lines[0]) if height > 0 else 0

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
                        slice_vectors.append(np.zeros(8))
            slice_array = np.array(slice_vectors).reshape(9, 8)
            affordance_data.append(slice_array)

    affordance_file_path = os.path.join(output_dir, output_name)
    np.save(affordance_file_path, np.array(affordance_data))
    print(f"Affordance data saved to {affordance_file_path}")

def chop_text_level(file_path, output_dir, output_name="chopped_text_level.npy", tile_size=3):
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
                    section.append("")
            chopped_sections.append(section)

    np_array = np.array(chopped_sections, dtype=object)
    output_path = os.path.join(output_dir, output_name)
    np.save(output_path, np_array)
    print(f"Chopped text level saved at {output_path}")

class TileProcessingApp:
    def __init__(self, master):
        self.master = master
        master.title("TileEmbedding Processing Tool")

        self.image_path = ""
        self.text_file_path = ""
        self.json_path = ""
        self.image_thumbnail = None

        # Image select and preview
        tk.Button(master, text="Select Image", command=self.select_image).pack(pady=5)
        self.image_label = tk.Label(master, text="No image selected", relief=tk.SUNKEN)
        self.image_label.pack(pady=5)

        tk.Button(master, text="Select LEVEL Text File", command=self.select_text_file).pack(pady=5)
        tk.Button(master, text="Select Affordance JSON File", command=self.select_json_file).pack(pady=5)

        tk.Button(master, text="Upscale Image", command=self.run_upscale).pack(pady=10)
        tk.Button(master, text="Downscale Image", command=self.run_downscale).pack(pady=10)
        tk.Button(master, text="Slice Image", command=self.run_slice).pack(pady=10)
        tk.Button(master, text="Chop Text Level Numpy", command=self.run_chop).pack(pady=10)
        tk.Button(master, text="Generate Affordance Numpy", command=self.run_affordance).pack(pady=10)

        # Status labels
        self.image_status = tk.Label(master, text="Image: None selected", anchor='w', fg='gray')
        self.image_status.pack(fill='x', padx=10)

        self.level_status = tk.Label(master, text="Level File: None selected", anchor='w', fg='gray')
        self.level_status.pack(fill='x', padx=10)

        self.affordance_status = tk.Label(master, text="Affordance File: None selected", anchor='w', fg='gray')
        self.affordance_status.pack(fill='x', padx=10)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.image_path:
            filename = os.path.basename(self.image_path)
            print(f"Selected image: {self.image_path}")
            self.image_status.config(text=f"Image: {filename}")
            self.show_image_preview()

    def show_image_preview(self):
        image = Image.open(self.image_path)
        image.thumbnail((300, 300))
        self.image_thumbnail = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_thumbnail, text="")

    def select_text_file(self):
        self.text_file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.text_file_path:
            filename = os.path.basename(self.text_file_path)
            print(f"Selected text file: {self.text_file_path}")
            self.level_status.config(text=f"Level File: {filename}")
            print("Text Level Loaded!")

    def select_json_file(self):
        self.json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if self.json_path:
            filename = os.path.basename(self.json_path)
            print(f"Selected JSON file: {self.json_path}")
            self.affordance_status.config(text=f"Affordance File: {filename}")
            print("Affordance JSON Loaded!")

    def run_upscale(self):
        if not self.image_path:
            messagebox.showwarning("Missing", "Please select an image first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Upscaled Image")
        if output_dir:
            upscale_image(self.image_path, output_dir)

    def run_downscale(self):
        if not self.image_path:
            messagebox.showwarning("Missing", "Please select an image first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Downscaled Image")
        if output_dir:
            downscale_image(self.image_path, output_dir)

    def run_slice(self):
        if not self.image_path:
            messagebox.showwarning("Missing", "Please select an image first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Sliced Images")
        if output_dir:
            slice_image(self.image_path, output_dir)

    def run_chop(self):
        if not self.text_file_path:
            messagebox.showwarning("Missing", "Please select a text file first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Chopped Text Level")
        if output_dir:
            chop_text_level(self.text_file_path, output_dir)

    def run_affordance(self):
        if not self.text_file_path or not self.json_path:
            messagebox.showwarning("Missing", "Please select a text file and JSON file first.")
            return
        output_dir = filedialog.askdirectory(title="Select Output Directory for Affordance Numpy")
        if output_dir:
            affordance_text_level(self.text_file_path, self.json_path, output_dir)

if __name__ == "__main__":
    root = tk.Tk()
    app = TileProcessingApp(root)
    root.mainloop()





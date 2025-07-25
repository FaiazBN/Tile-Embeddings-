import tkinter as tk
from tkinter import messagebox, colorchooser, filedialog
import json

affordance_mapping = [
    "breakable", "climbable", "collectable", "hazard",
    "moving", "passable", "portal", "solid"
]

character_data = {}


# --- Functions ---
def add_or_update_character():
    char = char_entry.get().strip()
    if not char:
        messagebox.showwarning("Warning", "Character is required.")
        return

    if not selected_color.get():
        messagebox.showwarning("Warning", "Color is required.")
        return

    selected_types = [aff for aff, var in type_vars.items() if var.get()]
    if not selected_types:
        messagebox.showwarning("Warning", "At least one type must be selected.")
        return

    character_data[char] = {
        "types": selected_types,
        "color": selected_color.get()
    }
    refresh_char_list()


def delete_character():
    selected = char_listbox.curselection()
    if not selected:
        return
    char = char_listbox.get(selected[0])
    if char in character_data:
        del character_data[char]
        refresh_char_list()


def choose_color():
    color_code = colorchooser.askcolor(title="Choose color")
    if color_code[1]:
        selected_color.set(color_code[1])


def refresh_char_list():
    char_listbox.delete(0, tk.END)
    for char in character_data:
        char_listbox.insert(tk.END, char)


def save_json_files():
    if not character_data:
        messagebox.showwarning("Warning", "No characters to save.")
        return

    # Save Types JSON
    types_json = {"tiles": {char: data["types"] for char, data in character_data.items()}}
    types_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")],
                                              title="Save Types JSON")
    if types_file:
        with open(types_file, 'w') as f:
            json.dump(types_json, f, indent=4)

    # Save Colors JSON
    color_json = {char: data["color"] for char, data in character_data.items()}
    colors_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")],
                                               title="Save Colors JSON")
    if colors_file:
        with open(colors_file, 'w') as f:
            json.dump(color_json, f, indent=4)

    messagebox.showinfo("Saved", "Files saved successfully.")


def on_char_select(event):
    selected = char_listbox.curselection()
    if not selected:
        return
    char = char_listbox.get(selected[0])
    data = character_data[char]

    char_entry.delete(0, tk.END)
    char_entry.insert(0, char)

    selected_color.set(data["color"])

    for aff, var in type_vars.items():
        var.set(aff in data["types"])


# --- GUI ---
root = tk.Tk()
root.title("JSON Maker Tool")

# Character input
tk.Label(root, text="Character:").grid(row=0, column=0, sticky='w')
char_entry = tk.Entry(root)
char_entry.grid(row=0, column=1, sticky='we')

# Color selection
tk.Label(root, text="Color:").grid(row=1, column=0, sticky='w')
selected_color = tk.StringVar()
tk.Entry(root, textvariable=selected_color).grid(row=1, column=1, sticky='we')
tk.Button(root, text="Pick Color", command=choose_color).grid(row=1, column=2, padx=5)

# Affordance checkboxes
tk.Label(root, text="Types:").grid(row=2, column=0, sticky='nw')
type_vars = {}
type_frame = tk.Frame(root)
type_frame.grid(row=2, column=1, sticky='w')

for aff in affordance_mapping:
    var = tk.BooleanVar()
    chk = tk.Checkbutton(type_frame, text=aff, variable=var)
    chk.pack(anchor='w')
    type_vars[aff] = var

# Buttons
tk.Button(root, text="Add / Update Character", command=add_or_update_character).grid(row=3, column=0, columnspan=2,
                                                                                     pady=5)
tk.Button(root, text="Delete Character", command=delete_character).grid(row=3, column=2)

# Character list
tk.Label(root, text="Characters:").grid(row=4, column=0, sticky='nw')
char_listbox = tk.Listbox(root, height=10)
char_listbox.grid(row=4, column=1, columnspan=2, sticky='we')
char_listbox.bind('<<ListboxSelect>>', on_char_select)

# Save button
tk.Button(root, text="Save JSON Files", command=save_json_files).grid(row=5, column=0, columnspan=3, pady=10)

root.columnconfigure(1, weight=1)
root.mainloop()
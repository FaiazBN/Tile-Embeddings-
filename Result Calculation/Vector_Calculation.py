import numpy as np
import pickle

# smk
smk_level_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/Result Calculation/Sliced Numpy/Super Mario Kart/smk1.npy"
smk_level_np = np.load(smk_level_dir, allow_pickle=True)
# solid
smk_solid_level_pickle_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Kart/Level 1/Slice_1764.pickle"
smk_solid_level_pickle = np.load(smk_solid_level_pickle_dir, allow_pickle=True)

# smb
smb_level_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/Result Calculation/Sliced Numpy/Super Mario Bros/smb1.npy"
smb_level_np = np.load(smb_level_dir, allow_pickle=True)

# solid
smb_solid_level_pickle_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Bros/Level 1/Slice_266.pickle"
smb_solid_level_pickle = np.load(smb_solid_level_pickle_dir, allow_pickle=True)
# passable
smb_passable_level_pickle_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Bros/Level 1/Slice_265.pickle"
smb_passable_level_pickle = np.load(smb_passable_level_pickle_dir, allow_pickle=True)

#print(smk_level_np)
#Super Mario Kart
for i in range(len(smk_level_np)):
    if smk_level_np[i][1][1] == '<' and len(smk_level_np[i][0]) == 3 and len(smk_level_np[i][1]) == 3 and len(smk_level_np[i][2]) == 3:
        print(smk_level_np[i])
        print(i+1)
# 1612, 1655, 1698
# 1786, 1805, 1764

#print(smk_level_np[379][1][1])
#Super Mario Bros Solid
# for i in range(len(smb_level_np)):
#     if len(smb_level_np[i][0]) != 3 or len(smb_level_np[i][1]) != 3 or len(smb_level_np[i][2]) != 3:
#         continue
#     if smb_level_np[i][1][1] == 'X':
#         print(smb_level_np[i])
#         print(i + 1)

#Super Mario Bros passable
# for i in range(len(smb_level_np)):
#     if len(smb_level_np[i][0]) != 3 or len(smb_level_np[i][1]) != 3 or len(smb_level_np[i][2]) != 3:
#         continue
#     if smb_level_np[i][1][1] == '-':
#         print(smb_level_np[i])
#         print(i + 1)


#difference vector
# difference_vector = smb_solid_level_pickle - smb_passable_level_pickle
#
# # smk new 1024:
# smk_new_pickle = smk_solid_level_pickle + difference_vector
# output_path = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/Result Calculation/smk_new_1764.pickle"
# with open(output_path, "wb") as f:
#     pickle.dump(smk_new_pickle, f)
#
#
#
# print("Shape: ", smk_new_pickle.shape)
#
# print("Difference vector shape:", difference_vector.shape)
# print(difference_vector)




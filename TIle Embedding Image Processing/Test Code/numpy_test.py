import numpy as np

# affordance_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/Numpy Data/Rainbow Islands/Level 8/affordance_data.npy"
# # text_level_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/Numpy Data/Super Mario Kart/Level 4/chopped_text_level.npy"
#
# # PCA_test_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Super Mario Bros_PCA_reduced.npy"
# normalized_test_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/Normalized/Super Mario Bros_PCA_reduced.npy"
# # VAE_Output_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/VAE Outputs/Super Mario Bros_PCA_reduced/Slice_1.pkl"
# original_slice_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/Super Mario Bros/Level 1/Slice_1.pickle"
# pca_inverse_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/TIle Embedding Image Processing/SSTE Data/PCA_Output/PCA_InverseTransformed/Super Mario Bros_PCA_inverse_transformed.pkl"
# pickle_pca_test = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/PCA/PCA_test_restored/Super Mario Kart/Slice_1.pickle"
# final_affordance_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/SSTE/decoded_texts.npy"
#slevel_text = np.load(text_level_dir,  allow_pickle=True)
# mapped_text = np.load(affordance_dir,  allow_pickle=True)

# # pca_file = np.load(PCA_test_dir, allow_pickle=True)
# normalized_file = np.load(normalized_test_dir, allow_pickle=True)
# # VAE_Output = np.load(VAE_Output_dir, allow_pickle=True)
# original_slice = np.load(original_slice_dir, allow_pickle=True)
# pca_inverse = np.load(pca_inverse_dir, allow_pickle=True)
# #final_aff = np.load(final_affordance_dir, allow_pickle=True)
# pickle_pca_test_dir = np.load(pickle_pca_test, allow_pickle=True)
# #print(mapped_text[88])
# # print(level_text[1848])
# #print(pca_file)
# #max_abs_values = np.max(np.abs(pca_file), axis=0)
# #print(max_abs_values)
# print()
# print()
# # print(original_slice.shape)
# # print(VAE_Output.shape)
# #print(normalized_file.shape)
# #print(normalized_file)
# #print(pca_inverse.shape)
# #print(final_aff)
# print(pickle_pca_test_dir.shape)
affordance_threshold_dir = "C:/Users/faiaz/OneDrive/Desktop/Research Stuffs/FAIAZ Tile Embeddings Stuff/SSTE/Results/Affordances/Super Mario Kart Solid 12.npy"
affordance_threshold = np.load(affordance_threshold_dir)
np.set_printoptions(suppress=True, precision=6)

print(affordance_threshold[4])
for i in range(len(affordance_threshold[0])):
    if affordance_threshold[4][i] > 0.7:
        #print(affordance_threshold[4][i])
        print(1, end=" ")
    else:
        print(0, end=" ")



import nbimporter
import numpy as np
import skimage
import multiprocess
import threading
import queue
import os,time
import math
import cv2
from p1 import get_visual_words

def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    '''
    HINTS:
    (1) We can use np.histogram with flattened wordmap
    '''
    hist, bin_edges = np.histogram(wordmap, dict_size)
    return hist / np.sum(hist)


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    '''
    HINTS:
    (1) Take care of Weights 
    (2) Try to build the pyramid in Bottom Up Manner
    (3) the output array should first contain the histogram for Level 0 (top most level) , followed by Level 1, and then Level 2.
    '''
    h, w = wordmap.shape
    L = layer_num - 1
    patch_width = math.floor(w / (2**L))
    patch_height = math.floor(h / (2**L))

    # At 0, you will have 1 histogram
    # At 1, you will have 4 histograms
    # At 2, you will have 16 histograms
    # We will sum up 4^l for each layer
    # Create an array to hold all the histograms
    hist_all = np.zeros(int(dict_size / 3 * (4**(L + 1)-1)))

    weights = []
    starts  = []
    ends    = []
    for l in range(L+1):
        # Precompute weights
        if l == 0 or l == 1:
            weight = 2 ** -L
        else:
            weight = 2 ** (l - L - 1)
        weights.append(weight)

        # Add one to the previous end
        if len(starts) == 0:
            start = 0
        else:
            start = ends[l - 1]
        starts.append(start)


        # The end will be 4^(l+1) * K - 1
        # The end is going to be the number of features * patches on the current level
        # So the number of features is: dict_size
        # The number of patches is 4^l:  1 for level 0, 4 for level 1, 16 for level 2
        ends.append(start + dict_size * 4 ** l)

    # Now let's loop over all the layers, starting with the finest layer
    # We can calculate the histograms then aggregate as we go
    for l in range(L, -1, -1):
    
        # Calculate start and end
        weight = weights[l]
        start  = starts[l]
        end    = ends[l]

        # Number of patches in the x and y direction for the current layer
        num_patches_x = (2 ** (l + 1))
        num_patches_y = (2 ** (l + 1))

        # Base case: we are handling layer L
        if end == len(hist_all):
            # Divide the wordmap into patches then get the features for each cell then concatenate all of the feature histograms
            patches    = [wordmap[i * patch_height : (i+1) * patch_height, j * patch_width : (j + 1) * patch_width] for i in range(0, 2 ** l) for j in range(0, 2 ** l)]
            hists_layer = [get_feature_from_wordmap(patch, dict_size) for patch in patches]
        # Else case: we are handling layer [0, L-1]
        else:
            # Get the histogram for the previous layer
            prev_start = starts[l + 1]
            prev_end   = ends[l + 1]

            # Holds onto all of the patches for the current layer
            hists_layer = []

            # Collect the patches for the current layer by aggregating the l+1 layer
            for i in range(0, num_patches_y, 2):
                for j in range(0, num_patches_x, 2):
                    # Get the 1d index into the histogram
                    idx_left_start  = prev_start + (i +     j * num_patches_y) * dict_size
                    idx_left_end    = prev_start + (i + 2 + j * num_patches_y) * dict_size
                    idx_right_start = prev_start + (i +     (j + 1) * num_patches_y) * dict_size
                    idx_right_end   = prev_start + (i + 2 + (j + 1) * num_patches_y) * dict_size

                    # The the group of four patches
                    hists = np.concatenate((hist_all[idx_left_start:idx_left_end], hist_all[idx_right_start:idx_right_end])).reshape(-1, dict_size)
                    hists = np.sum(hists, axis = 0)
                    hists_layer.append(hists)
                    
        # Once we have collected all the patches, we should concatenate them, normalize them, and multiply by the weight
        concat_hists = np.concatenate(hists_layer)
        hist_all[start:end] = concat_hists * (weight / np.sum(concat_hists))

    # Make sure the histograms are normalized to 1
    hist_all = hist_all / np.sum(hist_all)
    return hist_all


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    '''
    HINTS:
    (1) Consider A = [0.1,0.4,0.5] and B = [[0.2,0.3,0.5],[0.8,0.1,0.1]] then \
        similarity between element A and set B could be represented as [[0.1,0.3,0.5],[0.1,0.1,0.1]]   
    '''
    # Find the minimum between the word_histogram and the training histograms
    min_bin_values = np.minimum(word_hist, histograms)
    sim = np.sum(min_bin_values, axis = 1)
    return sim

def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # Load the image
    img = cv2.imread(file_path)
    
    # Get the wordmap from the image
    visual_words_image = get_visual_words(img, dictionary)

    # Get the features describing the wordmap in the form of a multilevel pyramid
    features = get_feature_from_wordmap_SPM(visual_words_image, layer_num, K)
    
    return [file_path, features]
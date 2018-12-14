# code for Scott- compute HOG feature vector for image

import numpy as np
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
from skimage.transform import resize

def FUNCTION_NAME(img_file):
    cell_len = 8

    # orientation bins for histogram
    orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
        (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]

    # 1. load image
    img = load_image(img_file)
    # 2. resize image to be 32 x 32
    resized_img = resize(img, (32, 32), anti_aliasing = True, mode = 'constant')

    # 3. set up feature vector, do some pre-processing on image, run canny_nmax()
    feature_vector = []
    resized_img = conv_1d_centered(resized_img)
    mag, theta = canny_nmax(resized_img)

    # 4. set up matrix of histograms for later block normalization
    hists = np.zeros((int(resized_img.shape[0] / cell_len), int(resized_img.shape[1] / cell_len), len(orient_bins) - 1))

    # 5. iterate through image to get 8x8 cells, compute histogram for each cell and store in hists
    for i in range(0, resized_img.shape[1], cell_len):
        for j in range(0, resized_img.shape[0], cell_len):
            mag_section = mag[j:j + cell_len, j:j + cell_len]
            theta_section = theta[j:j + cell_len, j:j + cell_len]
            hist = compute_cell_histogram(mag_section, theta_section, orient_bins)
            hists[int(j / cell_len), int(i / cell_len)] = hist

    # 6. perform block normalization on hists
    for i in range(hists.shape[1] - 1):
        for j in range(hists.shape[0] - 1):
            block_hist = np.concatenate((hists[i][j], hists[i + 1][j], hists[i][j + 1], hists[i + 1][j + 1]))
            try:
                normalized_block_hist = block_hist / np.linalg.norm(block_hist)
            except: # in case where np.linalg.norm(block_hist) == 0
                normalized_block_hist = block_hist
            feature_vector.append(normalized_block_hist)

    # 7. collapse feature_vector into one big array w/ 324 features
    feature_vector = np.concatenate(feature_vector)
    # len(feature_vector) == 324

    return feature_vector
import numpy as np

# HOGS (Histogram of Oriented Gradients) Person Detector (Dalal and Triggs, 2005)

# DESCRIPTION OF IMPLEMENTATION:
# - Following the original implementation, we use an image of 64 width x 128 height.
# - We operate on 8x8 cells within the detection window, organized into overlapping blocks.
# - Within a cell, we compute the gradient vector/histogram at each pixel, i.e. put the 64 gradient vectors
#   in our 8x8 cell into a 9-bin histogram ranging from 0 to 180 degrees with 20 degrees per bin. Each gradient's
#   contribution to the histogram is given by vector magnitude. We also split the contribution proportionally
#   between the two closest bins.
# - Next, histogram normalization using block normalization with 50 50 overlap. [Concatenate the histograms of
#   four cells within the block into a vector with 36 components (4 histograms x 9 bins )] Divide this vector
#   by its magnitude to normalize it.
# - Final size: the 64x128 detection window will be divided into 7 blocks across and 15 blocks vertical > 105 blocks.
#   Each block contains 4 cells with 9-bin histogram per cell.

'''
# FUNCTION DESCRIPTION
# Computes the gradient vector for a single 8x8 cell
# PARAMETERS
# - image: input image, which can be in color (?)
# - initial_x_coord: the horizontal offset (in multiples of 8) from the initial x coord of the sliding window.
# - initial_y_coord: the vertical offset (in multiples of 8) from the initial y coord of the sliding window.
# RETURNS
# - a 9-bin histogram, which is structured as a vector
'''
def compute_cell_histogram(y, x, mag_section, theta_section, orient_bins):
    # tterate through each pixel in the local 8x8 cell
    hist = [0] * 9
    for i in range(mag_section.shape[0]):
        for j in range(mag_section.shape[1]):
            if theta_section[j, i] == np.pi:
                ind = len(orient_bins) - 1
                adj_ind = 0
                # np.digitize interprets pi as falling outside the specified
                # orientation bins, so need to manually assign ind value
            else:
                ind = np.digitize(theta_section[j, i], orient_bins) - 1
                # split mag_section[j, i]nitude between two closest bins
                if theta_section[j, i] % (orient_bins[1] - orient_bins[0]) == 0:
                # direction falls perfectly between two bins
                    adj_ind = -1
                elif theta_section[j, i] % ((orient_bins[1] - orient_bins[0]) / 2) == 0:
                # direction falls perfectly in the center of a bin
                    adj_ind = -2
                elif theta_section[j, i] > (orient_bins[ind + 1] + orient_bins[ind]) / 2:
                    if ind == len(hist) - 1:
                        # wrap around to first bin
                        adj_ind = 0
                    else:
                        adj_ind = ind + 1
                elif theta_section[j, i] < (orient_bins[ind + 1] + orient_bins[ind]) / 2:
                    if ind == 0:
                        # wrap around to last bin
                        adj_ind = len(hist) - 1
                    else:
                        adj_ind = ind - 1
            
            try:
                if adj_ind == -1:
                    pct_split = 0.5
                    hist[ind] += pct_split * mag_section[j, i]
                    if ind == 0:
                        a_ind = len(hist) - 1
                    elif ind == len(hist) - 1:
                        a_ind = 0
                    else:
                        a_ind = ind - 1
                    hist[a_ind] += pct_split * mag_section[j, i]
                elif adj_ind == -2:
                    hist[ind] += mag_section[j, i]
                else:
                    if (adj_ind < ind) or (adj_ind == len(hist) - 1 and ind == 0): # account for case where bin wraps around end
                        pct_split = np.abs((orient_bins[ind + 1] - theta_section[j, i]) / (orient_bins[1] - orient_bins[0]))
                    else:
                        pct_split = np.abs((orient_bins[ind] - theta_section[j, i]) / (orient_bins[1] - orient_bins[0]))
                    hist[ind] += pct_split * mag_section[j, i]
                    hist[adj_ind] += (1 - pct_split) * mag_section[j, i]
            except Exception as e:
                print(e)
                continue
            # except: # need exception in case theta falls outside the specified bins (w/o try/except, program might crash)
    return hist


# FUNCTION DESCRIPTION
# Inserts each computed cell histogram into a two-dimensional matrix mirroring the image itself
# PARAMETERS
# - histogram_array: a pre-initialized histogram array for the entire image
# - histogram_cell: the histogram of the particular cell to be added
# RETURNS
# - an updated histogram array, with the new cell added in the correct location
# def insert_histogram_array():
    # TODO

# FUNCTION DESCRIPTION
# Normalizes the histograms using block normalization technique, where each "block" is 2 cells by 2 cells
# and each cell has dimensions 8x8
# PARAMETERS
# - histogram_array: the array of histograms for the entire image, saved in order of their locations
# RETURNS
# - a new histogram array, which contains normalized histograms
# def block_normalize(img_section):
    # TODO
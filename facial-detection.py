# HOGS (Histogram of Oriented Gradients) Person Detector
# Introduced by Dalal and Triggs at CVPR Conference in 2005

# DESCRIPTION OF IMPLEMENTATION:
# - Following the original implementation, we use an image of 64x128 pixels.
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

# - Fun fact: in a talk given by Dalal for ACM, he said he empirically determined that 6x6 cell size, with
#   3x3 block size, produced the lowest errors empirically.



# FUNCTION DESCRIPTION
# Computes the gradient vector for a single 8x8 cell
# PARAMETERS
# - image: input image, which can be in color (?)
# - initial_x_coord: the horizontal offset (in multiples of 8) from the initial x coord of the sliding window.
# - initial_y_coord: the vertical offset (in multiples of 8) from the initial y coord of the sliding window.
# RETURNS
# - a 9-bin histogram, which is structured as a vector of vectors (?)
def compute_cell_histogram(image, initial_x_coord, initial_y_coord):
    # TODO

# FUNCTION DESCRIPTION
# Inserts each computed cell histogram into a two-dimensional matrix mirroring the image itself
# PARAMETERS
# - histogram_array: a pre-initialized histogram array for the entire image
# - histogram_cell: the histogram of the particular cell to be added
# RETURNS
# - an updated histogram array, with the new cell added in the correct location
def insert_histogram_array():
    # TODO

# FUNCTION DESCRIPTION
# Normalizes the histograms using block normalization technique, where each "block" is 2 cells by 2 cells
# and each cell has dimensions 8x8
# PARAMETERS
# - histogram_array: the array of histograms for the entire image, saved in order of their locations
# RETURNS
# - a new histogram array, which contains normalized histograms
def block_normalize(histogram):
    # TODO


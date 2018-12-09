import numpy as np
import os
from skimage.transform import resize
# from sklearn.linear_model import LogisticRegression
from facial_detection_util import load_image, canny_nmax
from facial_detection import compute_cell_histogram

# orientation bins
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]


def create_training_x_y(cell_len = 8): # test diff cell sizes, since 8x8 was for human detection
    X_train = []
    y_train = []

    # positive training data (faces)
    for file in os.listdir('data/mit-cbcl-faces/train/face') + os.listdir('data/mit-cbcl-faces/train/non-face'):
        if file.endswith('.pgm'):
            # img_num = file[:-4]
            img_num = 'face00138'
            # convert image to grayscale (not nec. for training set, since all images are grayscale, but good to remember for testing)
            try:
                img = load_image('data/mit-cbcl-faces/train/face/' + img_num + '.pgm')
                y_train.append(1)
            except:
                img = load_image('data/mit-cbcl-faces/train/non-face/' + img_num + '.pgm')
                y_train.append(0)
            # training images are 19x19. resize to 24x24 to create a whole number of 8x8 cells
            img = resize(img, (24, 24), anti_aliasing = True, mode = 'constant')
            # get gradient magnitudes and orientations
            mag, theta = canny_nmax(img)
            for i in range(0, img.shape[0], cell_len):
                for j in range(0, img.shape[0], cell_len):
                    mag_section = mag[j:j + cell_len, j:j + cell_len]
                    theta_section = theta[j:j + cell_len, j:j + cell_len]
                    hist = compute_cell_histogram(mag_section, theta_section, orient_bins)



# FOR TESTING, NEED TO THINK ABOUT SCALE

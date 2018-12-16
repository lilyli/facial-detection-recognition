import numpy as np
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
import argparse
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from matplotlib.patches import Rectangle
from PIL import Image

window_w = 32
threshold = 0.7
cell_len = 8 #CHANGE IF NEED BE
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]

# EDIT UTIL HISTOGRAM FUNCTION TO DELETE TESTING PARAMETERS TO PRINT

# stick w/ 8x8 model, sample training images, retrain model w/ probabilities

'''
Given grayscale image, return the coordinates of the bounding boxes
of faces in the image

bbx coord: (y, x) for upper left hand corner of window
bbx[2]: prob of match
'''
def detect_faces(img, model):
    # slide 32 x 32 window over every pixel of image
    # apply svm to every window, check if prob > threshold

    img = conv_1d_centered(img)
    mag, theta = canny_nmax(img)
    X_test = []

    # iterate over every 32 x 32 section in the img
    for x in range(0, img.shape[1] - window_w, int(cell_len / 4)): # step by cell_len / 2 to speed up computation
        for y in range(0, img.shape[0] - window_w, int(cell_len / 2)):
            # calculate HOG feature vector for the 32 x 32 window
            X_test_feature = []
            hists = np.zeros((int(window_w / cell_len), int(window_w / cell_len), len(orient_bins) - 1))
            for i in range(x, x + window_w, cell_len):
                for j in range(y, y + window_w, cell_len):
                    mag_section = mag[j:j + cell_len, i:i + cell_len]
                    theta_section = theta[j:j + cell_len, i:i + cell_len]
                    hist = compute_cell_histogram(mag_section, theta_section, orient_bins)
                    hists[int((j - y) / cell_len), int((i - x) / cell_len)] = hist

            # perform block normalization on hists
            for i in range(hists.shape[1] - 1):
                for j in range(hists.shape[0] - 1):
                    block_hist = np.concatenate((hists[j][i], hists[j + 1][i], hists[j][i + 1], hists[j + 1][i + 1]))
                    try:
                        normalized_block_hist = block_hist / np.linalg.norm(block_hist)
                    except: # in case where np.linalg.norm(block_hist) == 0
                        normalized_block_hist = block_hist
                    X_test_feature.append(normalized_block_hist)

            X_test.append(np.concatenate(([y, x], np.concatenate(X_test_feature)), axis = 0))
    
    X_test = np.array(X_test)
    y_predictions = model.predict(X_test[:, 2:])
    X_test = X_test[np.where(y_predictions == 1)[0]][:, :2]
    return X_test


def detect_and_recognize_faces(img_list, model):
    # gray_img = load_image('data/detection-train/face/' + file)
    gray_img = load_image('test.jpg')
    faces_bbxs = []

    # for scale in (0.8, 1): #, 0.5, 1.1, 1.3, 1.5):
    scale = 0.3
    scaled_img = resize(gray_img, (round(gray_img.shape[0] * scale), round(gray_img.shape[1] * scale)), anti_aliasing = True, mode = 'constant')
    bbxs = detect_faces(scaled_img, model)
    for bbx in bbxs:
        faces_bbxs.append([bbx[0], bbx[1], window_w]) # don't have prob atm , bbx[2])

        # faces_bbxs.append([round(bbx[0] / scale), round(bbx[1] / scale), round(window_w / scale)]) # don't have prob atm , bbx[2])

    fig, ax = plt.subplots(1)
    ax.imshow(scaled_img) # gray_img)
    for bbx in faces_bbxs:
        rect = patches.Rectangle((bbx[1], bbx[0]), bbx[2], bbx[2],
            linewidth = 1, edgecolor = 'r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image for drawing boxes


    '''
    1: scale image to multiple scales. CHECK
    2. slide 32 x 32 window across img at each scale using detect_faces()
    3. collect all bounding boxes from all iterations of step 2 and scale to fit orig scale of img CHECK
    4. do non-max suppresion to delete duplicate boxes
    5. imshow() iamge with boxes highlighted
    '''


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description = 'Segment an image')
    # parser.add_argument('-images', type = str, nargs = '+', required = True,
    #     help = 'The images to segment. Seperate file names with a space.')
    # args = parser.parse_args()
    # img_files = vars(args)['images']
    model = pickle.load(open('linear_svm', 'rb'))

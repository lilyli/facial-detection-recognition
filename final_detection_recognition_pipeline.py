import numpy as np
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
import argparse
import sys
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

window_w = 32
threshold = 0.7
cell_len = 8 #CHANGE IF NEED BE
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]

# EDIT UTIL HISTOGRAM FUNCTION TO DELETE TESTING PARAMETERS TO PRINT


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
    for x in range(0, img.shape[1] - window_w, int(window_w / 2)): # step by window_w / 2 to speed up computation
        for y in range(0, img.shape[0] - window_w, int(window_w / 2)):
            # calculate HOG feature vector for the 32 x 32 window
            X_test_feature = []
            hists = np.zeros((int(window_w / cell_len), int(window_w / cell_len), len(orient_bins) - 1))
            for i in range(x, x + window_w, cell_len):
                for j in range(y, y + window_w, cell_len):
                    mag_section = mag[j:j + cell_len, j:j + cell_len]
                    theta_section = theta[j:j + cell_len, j:j + cell_len]
                    hist = compute_cell_histogram('xx', j, i, mag_section, theta_section, orient_bins)
                    hists[int((j - y) / cell_len), int((i - x) / cell_len)] = hist

            # perform block normalization on hists
            for i in range(hists.shape[1] - 1):
                for j in range(hists.shape[0] - 1):
                    block_hist = np.concatenate((hists[i][j], hists[i + 1][j], hists[i][j + 1], hists[i + 1][j + 1]))
                    normalized_block_hist = block_hist / np.linalg.norm(block_hist)
                    X_test_feature.append(normalized_block_hist) # ADD POSITIONS TO FEATURE VECTS

            X_test.append(np.concatenate(([y, x], np.concatenate(X_test_feature)), axis = 0))
            if y % 100 == 0:
                print('y', y)
        if x % 200 == 0:
            print('x', x)

            # run model on whoile of x_test, save those w/ prob above threshols
    
    X_test = np.array(X_test)
    y_predictions = model.predict(X_test[:, 2:])[:,0]

    X_test = X_test[~np.nonzero(y_predictions).any(axis=1)]

	return X_test[:, 2:]


def detect_and_recognize_faces(img_list):
	# gray_img = load_image('data/detection-train/face/' + file)
    img = load_image('test.jpg')
	faces_bbxs = []

	for scale in (0.5, 0.7, 0.9, 1): #, 1.1, 1.3, 1.5):
		scaled_img = resize(gray_img, (gray_img.shape[0] * scale, gray_img.shape[1] * scale), anti_aliasing = True, mode = 'constant')
		bbxs = detect_faces(scaled_img)
		for bbx in bbxs:
			faces_bbxs.append([round(bbx[0] / scale), round(bbx[1] / scale), round(window_w / scale)], bbx[2])

# https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image for drawing boxes


	'''
	1: scale iamge to multiple scales. CHECK
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
    model = pickle.load(open('linear_svm_8', 'rb'))

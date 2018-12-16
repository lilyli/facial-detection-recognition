import numpy as np
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
from facial_recognition_ldml_method import main
import argparse
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
import seaborn as sns


window_w = 32
cell_len = 8
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]


'''
Calculate the area of the intersection between two rectangles
r1 and r2 are np arrays or lists of len 4, in the coordinate order ymin, xmin, ymax, xmax
'''
def intersection_area(r1, r2):
    dy = min(max(r1[0], r1[2]), max(r2[0], r2[2])) - max(min(r1[0], r1[2]), min(r2[0], r2[2]))
    dx = min(max(r1[1], r1[3]), max(r2[1], r2[3])) - max(min(r1[1], r1[3]), min(r2[1], r2[3]))
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else: # the rectangles don't intersect
        return None


'''
Given grayscale image, return the coordinates of the bounding boxes
of faces in the image

bbx coord: (y, x) for upper left hand corner of window
'''
def detect_faces(img, model):
    # slide 32 x 32 window over every pixel of image, calculate
    # HOG feature vector and feed to trained facial detection SVM

    img = conv_1d_centered(img)
    mag, theta = canny_nmax(img)
    X_test = []

    # iterate over every 32 x 32 section in the img
    for x in range(0, img.shape[1] - window_w, int(cell_len / 2)): # step by cell_len / 2 to speed up computation
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
    y_distances = model.decision_function(X_test[:, 2:])

    X_test = X_test[np.where(y_predictions == 1)][:, :2]
    y_distances = y_distances[np.where(y_predictions == 1)]
    return X_test, y_distances


def nonmax_suppression_bbxs(bbxs, distances, img_size):
    # order boxes by distances from decision_function()
    order = np.argsort(-distances)
    distances = distances[order]
    bbxs = bbxs[order]

    # indicator vector
    keep_bbox = np.asarray([True] * len(distances))

    # overlap threshold above which the less confident detection is suppressed
    threshold = 0.5

    for i in range(len(bbxs)):
        cur_bbx = bbxs[i]
        cur_bbx_area = (cur_bbx[2] - cur_bbx[0]) * (cur_bbx[3] - cur_bbx[1])
        # check bbx hasn't already been suppressed
        if keep_bbox[i]:
            # iterate through other bbxs to find those who overlap
            for j in np.where(keep_bbox)[0]:
                if i == j: # don't examine pairs of the same bbx
                    other_bbx = bbxs[j]
                    other_bbx_area = (other_bbx[2] - other_bbx[0]) * (other_bbx[3] - other_bbx[1])

                    # check if the two boxes intersect
                    if ((cur_bbx[0] <= other_bbx[0] <= cur_bbx[0] + cur_bbx[2] \
                        or cur_bbx[0] <= other_bbx[2] <= cur_bbx[0] + cur_bbx[2]) \
                        and (cur_bbx[1] <= other_bbx[1] <= cur_bbx[1] + cur_bbx[3] \
                        or cur_bbx[1] <= other_bbx[3] <= cur_bbx[1] + cur_bbx[3])):
                        
                        intersect_area = intersection_area(cur_bbx, other_bbx)
                        # overlap = area of intersection / area of union
                        # suppress other boxes with an overlap >= threshold
                        if intersect_area / (cur_bbx_area + other_bbx_area) >= threshold:
                            keep_bbox[j] = False

    return np.where(keep_bbox)


def detect_and_recognize_faces(img_list, model):
    # all_faces = []
    # all_faces_bbx = []

    for img in img_list:
        gray_img = load_image(img)
        faces_bbxs = []

        for scale in [0.8, 0.9, 1, 1.1, 1.2]:
            scaled_img = resize(gray_img, (round(gray_img.shape[0] * scale),
                round(gray_img.shape[1] * scale)), anti_aliasing = True, mode = 'constant')
            bbxs, distances = detect_faces(scaled_img, model)
            for bbx in bbxs:
                faces_bbxs.append([round(bbx[0] / scale), round(bbx[1] / scale),
                    round(bbx[0] / scale) + round(bbx[0] / scale), round(bbx[0] / scale) + round(bbx[1] / scale)])

        valid_bbx_ind = nonmax_suppression_bbxs(np.array(faces_bbxs), distances, gray_img.shape)
        valid_bbxs = np.array(faces_bbxs)[np.where(valid_bbx_ind[0])]
        # for bbx in valid_bbxs:
        #     all_faces.append(scaled_img[bbx[0]:bbx[2], bbx[1]:bbx[3], img, (0, 0, 0)]) # add placeholder for color
        # all_faces_bbx.append(valid_bbxs)

        fig, ax = plt.subplots(1)
        ax.imshow(gray_img)
        for bbx in valid_bbxs:
            rect = patches.Rectangle((bbx[1], bbx[0]), bbx[2] - bbx[1], bbx[2] - bbx[1],
                linewidth = 1, edgecolor = 'r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    # colors = sns.color_palette('bright', len(all_faces))
    # ldml_model = main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Segment an image')
    parser.add_argument('-images', type = str, nargs = '+', required = True,
        help = 'The images to segment. Seperate file names with a space.')
    args = parser.parse_args()
    img_files = vars(args)['images']
    model = pickle.load(open('linear_svm', 'rb'))
    detect_and_recognize_faces(img_files, model)


import numpy as np
import os
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
from sklearn import svm
import pickle

# orientation bins
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]


'''
    Create training set data

    Returns:
        X_train (np array) - features of training set
        y_train (np array) - labels of training set
'''
def create_training_x_y(cell_len = 8): # test diff cell sizes, since 8x8 was for human detection
    X_train = []
    y_train = []

    x = 0

    # positive training data (faces)
    for file in os.listdir('data/detection-train/face') + os.listdir('data/detection-train/non-face'):
        if file.endswith('.png'):
            try:
                img = load_image('data/detection-train/face/' + file)
                y_train.append(1)
            except:
                img = load_image('data/detection-train/non-face/' + file)
                y_train.append(0)

            x += 1
            if x % 200 == 0:
                print(x)

            X_train_feature = []

            img = conv_1d_centered(img)

            hists = np.zeros((int(img.shape[0] / cell_len), int(img.shape[1] / cell_len), len(orient_bins) - 1))
            # get gradient magnitudes and orientations, compute histograms and store in hists
            mag, theta = canny_nmax(img)
            for i in range(0, img.shape[0], cell_len):
                for j in range(0, img.shape[0], cell_len):
                    # print(i, j)
                    mag_section = mag[j:j + cell_len, j:j + cell_len]
                    theta_section = theta[j:j + cell_len, j:j + cell_len]
                    hist = compute_cell_histogram(file, j, i, mag_section, theta_section, orient_bins)
                    hists[int(j / cell_len), int(i / cell_len)] = hist

            # perform block normalization on hists
            for i in range(hists.shape[1] - 1):
                for j in range(hists.shape[0] - 1):
                    block_hist = np.concatenate((hists[i][j], hists[i + 1][j], hists[i][j + 1], hists[i + 1][j + 1]))
                    normalized_block_hist = block_hist / np.linalg.norm(block_hist)
                    X_train_feature.append(normalized_block_hist)

            X_train.append(np.concatenate(X_train_feature, axis = 0))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train

# FOR TESTING, NEED TO THINK ABOUT SCALE OF IMAGE- resample image at multiple scales
# might want to train w/ labelled faces in wild dataset instead, can use both color and b&w images


'''
    Train the model

    Returns:
        model (sklearn model) - SVM model that predicts the probability
            an image is a face
'''
if __name__ == '__main__':
    X_train, y_train = create_training_x_y()
    pickle.dump(X_train, open('X_train_8', 'wb'))
    pickle.dump(y_train, open('y_train_8', 'wb'))
    # X_train_1, y_train_1 = create_training_x_y(cell_len = 4)
    #create and train an svm classifier
    model = svm.SVC(kernel='linear')
    # model_1 = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    # model_1.fit(X_train_1, y_train_1)
    print("Training accuracy w/ cell length = 8:", model.score(X_train, y_train))
    # print("Training accuracy w/ cell length = 4:", model_1.score(X_train_1, y_train_1))
    pickle.dump(model, open('linear_svm_8', 'wb'))


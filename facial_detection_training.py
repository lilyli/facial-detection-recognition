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
            mag, theta = canny_nmax(img)
            hists = np.zeros((int(img.shape[0] / cell_len), int(img.shape[1] / cell_len), len(orient_bins) - 1))

            # get gradient magnitudes and orientations, compute histograms and store in hists
            for i in range(0, img.shape[1], cell_len):
                for j in range(0, img.shape[0], cell_len):
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
    # remove non-finite values (e.g. NaN) that may have arisen
    X_train_1 = X_train[~np.isnan(X_train).any(axis=1)]
    y_train = np.array(y_train)[~np.isnan(X_train).any(axis=1)]

    return X_train_1, y_train

# FOR TESTING, NEED TO THINK ABOUT SCALE OF IMAGE- resample image at multiple scales


'''
    Train the model

    Returns:
        model (sklearn model) - SVM model that predicts the probability
            an image is a face
'''
if __name__ == '__main__':
    # train model w/ cell len = 8
    # X_train, y_train_8 = create_training_x_y()
    X_train = pickle.load(open('X_train_8', 'rb'))
    y_train_8 = pickle.load(open('y_train_8', 'rb'))
    X_train_8 = X_train[~np.isnan(X_train).any(axis=1)]
    y_train_8 = np.array(y_train_8)[~np.isnan(X_train).any(axis=1)]
    # pickle.dump(X_train_8, open('X_train_8', 'wb'))
    # pickle.dump(y_train_8, open('y_train_8', 'wb'))
    # create and train an svm classifier
    model_8 = svm.SVC(kernel='linear', probability = True)
    model_8.fit(X_train_8, y_train_8)
    print("Training accuracy w/ cell len = 8:", model_8.score(X_train_8, y_train_8))
    # 0.7570450334326916
    pickle.dump(model_8, open('linear_svm_8_proba_true', 'wb'))

# FINAl image counts: 24058 training faces, 38455 non-faces
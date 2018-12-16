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

    # positive training data (faces)
    for file in os.listdir('data/detection-train/face') + os.listdir('data/detection-train/non-face'):
        if file.endswith('.png'):
            try:
                img = load_image('data/detection-train/face/' + file)
                y_train.append(1)
            except:
                img = load_image('data/detection-train/non-face/' + file)
                y_train.append(0)

            X_train_feature = []
            img = conv_1d_centered(img)
            mag, theta = canny_nmax(img)
            hists = np.zeros((int(img.shape[0] / cell_len), int(img.shape[1] / cell_len), len(orient_bins) - 1))

            # get gradient magnitudes and orientations, compute histograms and store in hists
            for i in range(0, img.shape[1], cell_len):
                for j in range(0, img.shape[0], cell_len):
                    mag_section = mag[j:j + cell_len, i:i + cell_len]
                    theta_section = theta[j:j + cell_len, i:i + cell_len]
                    hist = compute_cell_histogram(mag_section, theta_section, orient_bins)
                    hists[int(j / cell_len), int(i / cell_len)] = hist

            # perform block normalization on hists
            for i in range(hists.shape[1] - 1):
                for j in range(hists.shape[0] - 1):
                    block_hist = np.concatenate((hists[j][i], hists[j + 1][i], hists[j][i + 1], hists[j + 1][i + 1]))
                    try:
                        normalized_block_hist = block_hist / np.linalg.norm(block_hist)
                    except: # in case where np.linalg.norm(block_hist) == 0
                        normalized_block_hist = block_hist
                    X_train_feature.append(normalized_block_hist)

            X_train.append(np.concatenate(X_train_feature, axis = 0))

    X_train = np.array(X_train)
    # remove non-finite values (e.g. NaN) that may have arisen
    X_train_1 = X_train[~np.isnan(X_train).any(axis=1)]
    y_train = np.array(y_train)[~np.isnan(X_train).any(axis=1)]

    return X_train_1, y_train


'''
    Train the model

    Returns:
        model (sklearn model) - SVM model that predicts the probability
            an image is a face
'''
if __name__ == '__main__':
    # X_train, y_train = create_training_x_y()
    # pickle.dump(X_train, open('X_train', 'wb'))
    # pickle.dump(y_train, open('y_train', 'wb'))
    X_train = pickle.load(open('X_train', 'rb'))
    y_train = pickle.load(open('y_train', 'rb'))

    # create and train an svm classifier
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)
    print("Training accuracy of linear SVM:", model.score(X_train, y_train))
    pickle.dump(model, open('linear_svm', 'wb'))

    model_2 = svm.LinearSVC(C = 0.0001)
    model_2.fit(X_train, y_train)
    print("Training accuracy of linear SVC w/ C = 0.0001:", model_2.score(X_train, y_train))
    pickle.dump(model_2, open('linear_svc', 'wb'))

    model_3 = svm.LinearSVC(C = 0.01)
    model_3.fit(X_train, y_train)
    print("Training accuracy of linear SVC w/ C = 0.01:", model_3.score(X_train, y_train))
    pickle.dump(model_3, open('linear_svc_1', 'wb'))


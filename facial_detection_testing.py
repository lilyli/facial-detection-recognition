import numpy as np
import os
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

# orientation bins
orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
    (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]


'''
    Create testing set data

    Returns:
        X_test (np array) - features of testing set
        y_test (np array) - labels of testing set
'''
def create_testing_x_y(cell_len = 8): # test diff cell sizes, since 8x8 was for human detection
    X_test = []
    y_test = []

    x = 0

    # positive testing data (faces)
    for file in os.listdir('data/detection-test/face') + os.listdir('data/detection-test/non-face'):
        if file.endswith('.png'):
            try:
                img = load_image('data/detection-test/face/' + file)
                y_test.append(1)
            except:
                img = load_image('data/detection-test/non-face/' + file)
                y_test.append(0)

            x += 1
            if x % 200 == 0:
                print(x)

            X_test_feature = []
            img = conv_1d_centered(img)
            mag, theta = canny_nmax(img)
            hists = np.zeros((int(img.shape[0] / cell_len), int(img.shape[1] / cell_len), len(orient_bins) - 1))

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
                    X_test_feature.append(normalized_block_hist)

            X_test.append(np.concatenate(X_test_feature, axis = 0))

    X_test = np.array(X_test)
    # remove non-finite values (e.g. NaN) that may have arisen
    X_test_1 = X_test[~np.isnan(X_test).any(axis=1)]
    y_test = np.array(y_test)[~np.isnan(X_test).any(axis=1)]

    return X_test_1, y_test


'''
    Test the model
'''
if __name__ == '__main__':
    # test model w/ cell len = 8
    X_test_8, y_test_8 = create_testing_x_y()
    model_8 = pickle.load(open('linear_svm_8', 'rb'))
    predictions_8 = model_8.predict(X_test_8)
    print('Accuracy score w/ cell len = 8:', accuracy_score(y_test_8, predictions_8))
    precision_8, recall_8, fscore_8, support_8 = score(y_test_8, predictions_8)
    print('Precision: {}'.format(precision_8))
    print('Recall: {}'.format(recall_8))
    print('F-score: {}'.format(fscore_8))
    print('Support: {}'.format(support_8))    

    # FINAl image counts: 4996 testing faces, 7787 non-faces
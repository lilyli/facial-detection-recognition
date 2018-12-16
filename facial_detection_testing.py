import numpy as np
import os
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
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

    # positive testing data (faces)
    for file in os.listdir('data/detection-test/face') + os.listdir('data/detection-test/non-face'):
        if file.endswith('.png'):
            try:
                img = load_image('data/detection-test/face/' + file)
                y_test.append(1)
            except:
                img = load_image('data/detection-test/non-face/' + file)
                y_test.append(0)

            X_test_feature = []
            img = conv_1d_centered(img)
            mag, theta = canny_nmax(img)
            hists = np.zeros((int(img.shape[0] / cell_len), int(img.shape[1] / cell_len), len(orient_bins) - 1))

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
    # X_test, y_test = create_testing_x_y()
    # pickle.dump(X_test, open('X_test', 'wb'))
    # pickle.dump(y_test, open('y_test', 'wb'))
    X_test = pickle.load(open('X_test', 'rb'))
    y_test = pickle.load(open('y_test', 'rb'))
    model_1 = pickle.load(open('linear_svm', 'rb'))
    model_2 = pickle.load(open('linear_svc', 'rb'))
    model_3 = pickle.load(open('linear_svc_1', 'rb'))

    for model in (model_1, model_2, model_3):
        predictions = model.predict(X_test)
        # print('Accuracy score for linear SVM:', accuracy_score(y_test, predictions))
        print('Accuracy score for linear SVC w/ C = 0.01:', accuracy_score(y_test, predictions))
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, predictions)
        print('Precision: {}'.format(precision))
        print('Recall: {}'.format(recall))
        print('F-score: {}'.format(fscore))
        print('Support: {}'.format(support))


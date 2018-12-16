import os
import numpy as np
from scipy.spatial import distance
from facial_detection_util import load_image, canny_nmax, conv_1d_centered, compute_cell_histogram
from skimage.transform import resize

#################### Implementation of LDML Classifier #######################

'''
    Sigmoid transformation for logistic model construction

    Returns:
        int - sigmoid transformation of input 
'''
def sigmoid(input):
    return 1 / (1 + np.exp(-input))


'''
    Return weights of logistic model, WITHOUT intercept/biased term for now

    Returns:
        ws - weights 
'''
def train_weights(gt_dist, gt_labels, num_steps, learning_rate):
    # we are optimizing for gt_dist and the intercept (bias) term, so create a new features container
    gt_dist = np.array(gt_dist)
    gt_labels = np.array(gt_labels)
    features = gt_labels
    # initialize weights
    weights = np.zeros(shape=features.shape)
    for epoch in range(num_steps):
        scores = np.dot(features, weights) # check correlation between feature vector and weight
        sigmoid_output = sigmoid(scores) # apply sigmoid to get logistic curve
        loss = gt_labels - sigmoid_output # calculate loss between ground truth and predicted sigmoid
        gradient = np.dot(features.T, loss) # calculate rate of change of features vector w.r.t. loss function
        weights += learning_rate * gradient
    return weights

'''
    Extracts feature vector for an image

    Returns:
        feature_vector - a feature vector with 324 features 
'''
def extract_feature_vector(img_file):
    cell_len = 8
    # orientation bins for histogram
    orient_bins = [0, np.pi / 9, (2 * np.pi) / 9, np.pi / 3, (4 * np.pi) / 9,
        (5 * np.pi) / 9, (2 * np.pi) / 3, (7 * np.pi) / 9, (8 * np.pi) / 9, np.pi]
    # 1. load image
    img = load_image(img_file)
    # 2. resize image to be 32 x 32
    resized_img = resize(img, (32, 32), anti_aliasing = True, mode = 'constant')
    # 3. set up feature vector, do some pre-processing on image, run canny_nmax()
    feature_vector = []
    resized_img = conv_1d_centered(resized_img)
    mag, theta = canny_nmax(resized_img)
    # 4. set up matrix of histograms for later block normalization
    hists = np.zeros((int(resized_img.shape[0] / cell_len), int(resized_img.shape[1] / cell_len), len(orient_bins) - 1))
    # 5. iterate through image to get 8x8 cells, compute histogram for each cell and store in hists
    for i in range(0, img.shape[1], cell_len):
        for j in range(0, img.shape[0], cell_len):
            mag_section = mag[j:j + cell_len, i:i + cell_len]
            theta_section = theta[j:j + cell_len, i:i + cell_len]
            hist = compute_cell_histogram(mag_section, theta_section, orient_bins)
            hists[int(j / cell_len), int(i / cell_len)] = hist
    # 6. perform block normalization on hists
    for i in range(hists.shape[1] - 1):
        for j in range(hists.shape[0] - 1):
            block_hist = np.concatenate((hists[j][i], hists[j + 1][i], hists[j][i + 1], hists[j + 1][i + 1]))
            try:
                normalized_block_hist = block_hist / np.linalg.norm(block_hist)
            except: # in case where np.linalg.norm(block_hist) == 0
                normalized_block_hist = block_hist
            feature_vector.append(normalized_block_hist)
    # 7. collapse feature_vector into one big array w/ 324 features
    feature_vector = np.concatenate(feature_vector)
    # len(feature_vector) == 324
    return feature_vector


'''
    Create training set data

    Returns:
        x_train - distances from training set
        y_train - labels from training set
'''
def make_training_data(lines, label_type):
    print("Making training data...")
    dir_working = str(os.getcwd())
    dir_train = dir_working + "\\data\\recognition-train\\faces\\"
    dist_array = []
    label_array = []
    i = 0
    for idx in range(len(lines)):
        first = lines[idx].partition(' ')[0]
        second = lines[idx].partition(' ')[2]

        for img in os.listdir(dir_train):
            if img == str(first + ".pgm"):
                for img_alt in os.listdir(dir_train):
                    if img_alt == str(second + ".pgm"):
                        path = dir_train + img
                        path_alt = dir_train + img_alt
                        vt_0 = extract_feature_vector(path)
                        vt_1 = extract_feature_vector(path_alt)
                        # instead of inverse, we use the Moore-Penrose inverse, in case we encounter singular matrices.
                        # this should be OK for calculating Mahalanobis matrix.
                        # per https://stats.stackexchange.com/questions/37743/singular-covariance-matrix-in-mahalanobis-distance-in-matlab
                        dist = distance.euclidean(vt_0, vt_1)
                        dist_array.append(dist)
                        if label_type == 0:
                            label_array.append(0)
                        else:
                            label_array.append(1)
                        break
                    else:
                        continue
            else:
                continue
        i += 1
        if i == 20:
            break
    x_train = dist_array
    y_train = label_array
    return x_train, y_train

def execute_test(lines, threshold, trained_ws):
    print("Executing test...")
    dir_working = str(os.getcwd())
    dir_test = dir_working + "\\data\\recognition-train\\faces\\"
    label_array = []
    i = 0
    for idx in range(len(lines)):
        first = lines[idx].partition(' ')[0]
        second = lines[idx].partition(' ')[2]
        for img in os.listdir(dir_test):
            if img == str(first + ".pgm"):
                for img_alt in os.listdir(dir_test):
                    if img_alt == str(second + ".pgm"):
                        path = dir_test + img
                        path_alt = dir_test + img_alt
                        vt_0 = extract_feature_vector(path)
                        vt_1 = extract_feature_vector(path_alt)
                        dist = distance.euclidean(vt_0, vt_1)
                        features = [dist]
                        probability = sigmoid(np.dot(features, trained_ws[0])) # TODO: limitation is only one dimension
                        if probability > threshold:
                            label_array.append(1)
                        else:
                            label_array.append(0)
                        break
                    else:
                        continue
            else:
                continue
        i += 1
        if i == 20:
            break
    y_test = label_array
    return y_test

'''
    Produce weights from LDML classifier

    Returns:
        ws - weights 
'''
def train_ldml():
    print("Training LDML model...")
    dir_working = str(os.getcwd())
    dir_file = dir_working + "\\data\\recognition-train\\lists"
    with open(dir_file+"\\01_train_same.txt", "r") as f_same:
        lines_same = f_same.read().splitlines()
    with open(dir_file + "\\01_train_diff.txt", "r") as f_diff:
        lines_diff = f_diff.read().splitlines()
    x_train_same, y_train_same = make_training_data(lines_same, 1)
    x_train_diff, y_train_diff = make_training_data(lines_diff, 0)
    x_train = x_train_same + x_train_diff
    y_train = y_train_same + y_train_diff
    ws = train_weights(x_train, y_train,300000, 5e-5)
    return ws

def test_ldml(trained_ws):
    print("Testing LDML model...")
    dir_working = str(os.getcwd())
    dir_file = dir_working + "\\data\\recognition-train\\lists"
    with open(dir_file + "\\02_test_same.txt", "r") as f_same:
        lines_same = f_same.read().splitlines()
    with open(dir_file + "\\02_test_diff.txt", "r") as f_diff:
        lines_diff = f_diff.read().splitlines()
    y_test_same = execute_test(lines_same, 0.7, trained_ws)
    y_test_diff = execute_test(lines_diff, 0.7, trained_ws)

    true_positives = 0
    for i in y_test_same:
        if i == 1:
            true_positives += 1

    false_positives = 0
    for i in y_test_diff:
        if i == 1:
            false_positives += 1

    false_negatives = 0
    for i in y_test_same:
        if i == 0:
            false_negatives += 1

    print(true_positives)
    print(false_positives)
    print(false_negatives)
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    return precision, recall


def main():
    weights = train_ldml()
    print("Weights are " + str(weights))
    p, r = test_ldml(weights)
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    return weights


if __name__ == "__main__":
    main()
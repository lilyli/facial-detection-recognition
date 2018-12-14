import os
import numpy as np
from scipy.spatial import distance

#################### Implementation of LDML Classifier #######################

# An implementation of the LDML technique in Guillaumin et. al (2009). This technique
# learns a linear transformation by optimizing the parameters of a logistic discriminant model,
# which is used to determine if two images depict the same person or not.
# P(yi = yj | xi, xj, M, b) = (1 + exp(dM(xi, xj) - b))^-1
# where yi and yj are images, xi and xj are feature vectors of those images, dM is Mahalanobis distance, b a bias term.
# Both M and b are learned using gradient ascent on the log-likelihood of P(yi = yj | xi, xj, M, b).

# Function for sigmoid linking
def sigmoid(input):
    return 1 / (1 + np.exp(-input))

# Function to calculate Mahalanobis Distance for a pair of images
def mahalanobis_dist(vector0, vector1, inverse_of_covariance_mx):
    dist = distance.mahalanobis(vector0, vector1, np.linalg.inv(np.cov(vector1,vector1))) # how to find inverse of covariance?
    return dist

# Build logistic model using gradient ascent, with intercept term
# Need gradient ascent because the likelihood method for building logistic model does not have closed-form solution
# http://cs.wellesley.edu/~sravana/ml/logisticregression.pdf
# https://beckernick.github.io/logistic-regression-from-scratch/
# and OH with Dr. Maire.
def train_weights(gt_dist, gt_labels, num_steps, learning_rate):
    # we are optimizing for gt_dist and the intercept (bias) term, so create a new features container
    intercept = np.ones((gt_dist.shape[0]), 1)
    features = np.hstack((intercept, gt_dist))
    # initialize weights
    weights = np.zeros(features.shape[1])

    for epoch in xrange(num_steps):
        scores = np.dot(features, weights) # check correlation between feature vector and weight
        sigmoid_output = sigmoid(scores) # apply sigmoid to get logistic curve
        loss = gt_labels - sigmoid_output # calculate loss between ground truth and predicted sigmoid
        gradient = np.dot(features.T, loss) # calculate rate of change of features vector w.r.t. loss function
        weights += learning_rate * gradient

    return weights

# TODO: update this temporary placeholder with HOG feature descriptor
def extract_feature_vector(img):
    vector = img[0] # temporary
    return vector

def train_LDML():
    dir_working = str(os.getcwd())
    dir_train = dir_working + "\\data\\<NAME OF SUB FOLDER WITH TRAINING IMAGES>"
    gtlabels_file = dir_working + "\\data\\<NAME OF <<FILE>> WITH GROUND TRUTH LABELS>"

    # Assuming that the training image folder is not broken down into further folders, i.e. images are there as-is.
    # for first gt img, iterate through 2nd, 3rd, etc. to find gt dist. Then put together array of gt_dist's.
    # for second gt img, iterate through 3rd, 4th, etc. to find gt dist. Then add to array of gt_dist's.
    # for each gt img, do similarly to find array for gt_label's.

    list_of_files = os.listdir(dir_train)
    list_of_images = []
    for file in list_of_files:
        if file.endswith("<INSERT FORMAT OF IMAGE>"):
            list_of_images.append(file)
        else:
            continue

    dist_mx = np.zeros(shape=(len(list_of_images), len(list_of_images)))
    gtlabel_mx = np.zeros(shape=(len(list_of_images), len(list_of_images)))

    base_id = 0
    for base in list_of_images:
        base = base + dir_train + "\\" + base
        base_vt = extract_feature_vector(base) # feature vector of base image
        toskip = base_id
        companion_id, images_skipped = 0
        for companion in list_of_images:
            if toskip == images_skipped:
                print("Differencing img with base id" + str(base_id) + "with that of companion id " + str(companion_id))
                companion_vt = extract_feature_vector(companion)

                d = mahalanobis_dist(base_vt, companion_vt, np.linalg.inv(np.cov(base_vt,companion_vt)))
                dist_mx[base_id][companion_id] = d # TODO: convert this to np.array to accommodate additional dimensions!

                label = resolve_gtlabel(gtlabels_file, base_id, companion_id)
                gtlabel_mx[base_id][companion_id] = label
            elif images_skipped < toskip:
                images_skipped += 1
            else:
                raise Exception("images_skipped should not exceed toskip")
            companion_id += 1
        base_id += 1

        # TODO: Still need to trim away nonzeros
        # TODO: Add another dimension
        dists = np.matrix.flatten(dist_mx)
        labels = np.matrix.flatten(gtlabel_mx)

        # can possibly add more dimensions of features, not just Mahalanobis distance
        ws = train_weights(dists, labels, 1000, 0.01) # last two parameters were arbitrary
        return ws

def resolve_gtlabel(file, base_idx, companion_idx):
    # TODO: instructions below
    # find the gtlabel0 of the image with index base_idx
    # find the gtlabel1 of the image with index companion_idx
    # if gtlabel0 == gtlabel1, then return 1
    # if gtlabel0 != gtlabel1, then return 0
    return 0

# Predicts LDML probability, using trained weights, for a single image.
# Given an feature vector (now just Mahalanobis dist, potentially more), and trained weights, return probability
# that this distance vector belongs to a pair of images that depict the same face.
def predict_ldml_singlepair(img0, img1, trained_ws):
    features0 = extract_feature_vector(img0)
    features1 = extract_feature_vector(img1)
    d = mahalanobis_dist(features0, features1, np.linalg.inv(np.cov(features0,features1)))
    features = [d] # can potentially append more features, in addition to Mahalanobis distance
    return sigmoid(np.dot(features, trained_ws))

def test_LDML():
    res0 = predict_ldml_singlepair()
    # TODO: For testing, calculate the differences from a bunch of images, and apply the predict() function
    # TODO: check by inspection (?)
    # TODO: check against the gt in the test dataset
    # TODO: returns a numeric saying how many percentage passed
    positive_count = -1
    all_count = -1
    precision = positive_count / all_count # TODO: update these count values based on iteration through test set
    recall = positive_count / actual_positive_count
    f_score= 2 * ( (precision * recall) / (precision + recall) )
        return precision, recall, f_score

if __name__ == "__main__":
    weights = train_LDML()
    p, r, f = test_LDML()
    print("Testing results are below...")
    print("Precision: " + str(p))
    print("Recall: " + str(r))
    print("F-Score: " + str(f))

#################### Implementation of mKnn Classifier #######################

# An implementation of the mKnn classifier as per Guillaumin et al. (2009). mKNN stands for Marginalized KNN,
# which means taking the marginal probability of xi and xj belonging to the same class.
# Google Books reference: https://tinyurl.com/y9kh3b9m

# The purpose of Knn is to use a database in which the data points are separated into different classes to predict
# the classification of a new sample point.

# Step 1: Find the nearest neighbor for each image.
# returns the histogram of class counts, for every image
def find_nearest_neighbor(k, list_of_images, list_of_labels):

    all_histograms = []
    index = 0
    for img in list_of_images:
        dists = []
        vt0 = extract_feature_vector(img)
        for id in range(len(list_of_images)):
            if id == index: # ignore the image vector itself
                continue
            else:
                d = distance.euclidean(vt0, list_of_images[id])
                dists.append((list_of_images[id], d, id)) # appends a three-element tuple
        dists.sort(key=operator.itemgetter(1))
        neighbors = []
        for id in range(k):
            neighbors.append(dists[id][2]) # appending the "id", within the list_of_images, that was added

        sorted_values = np.unique(neighbors) # this will be indices of the histogram
        num_bins = len(sorted_values)
        histogram = np.zeros(shape=(num_bins, 2))

        for id in range(num_bins):
            histogram[id][0] = sorted_values[id] # creating the "labels" of the bins of the histogram

        for id in neighbors:
            class_id = list_of_labels[id]
            bin_id = sorted_values.index(class_id) # find the index of an item given a list containing it
            histogram[bin_id][1] += 1 # incrementing the "bin" part of the bins of the histogram
            # now we have a histogram that tallies counts of each class in its neighboring region, for this image

        all_histograms.append(histogram) # the order is preserved in terms of "for img in list_of_images"
        index += 1
    return all_histograms

# returns, for an image pair, the probability metric
def probability_per_pair(histogram_0, histogram_1, k):
    bin_labels_0 = histogram_0[:,0] # extract the 1st column (labels of histogram bins) as new list
    bin_labels_1 = histogram_1[:,0]

    pairs = []
    for label_0 in bin_labels_0:
        for label_1 in bin_labels_1:
            if label_0 != label_1:
                continue
            else:
                product = histogram_0[label_0] * histogram_1[label_1] # multiple the counts of this class in histograms 0 and 1
                pairs.append(product)
                break
    numerator = sum(pairs)
    denominator = k**2
    probability = numerator / denominator # probability that the class labels of the two images (histograms) are equal
    return probability

def train_mknn_using_lmnn():
    # Can't use LDML to train, but we need LMNN which is optimized for kNN, as mentioned in the paper.
    # TODO: use the pyLMNN package
    # TODO: clarify with professor - the relationship between MkNN and LMNN

def predict_mknn_singlepair(target_img_0, target_img_1, expanded_list_of_imgs):
    # TODO: how to test a "new" pair? How to compute the nearest neighbors of this "New" pair? Introduce to existing KNN reservoir?


# Step 5: <Decide a threshold for probability to convert from a number between 0<x<1 to binary classifier output?>



    #################### Questions #######################

# Q1 - general for both LDML and KNN
# (a) How did the authors derive the feature vectors of their images xi and xj? It sounds like their paper presents
# the training schemes, which are dependent on the feature vectors being meaningful vector summaries of images.
# (b) How to use the Labelled Faces in the Wild data set? Is it downloadable? The downloadable seems to be a .txt file.
# See http://vis-www.cs.umass.edu/lfw/?fbclid=IwAR1TFXdZb0zA6i_TcuQOx6tgrNPESChoC_C6RWNwf2fsWEB2IxkLysghAlI#views.

# Q2 - What are the weights for the logistic classifier? What is the gradient ascent learning? The intercept (or
# bias term) just means adding another empty feature column, correct?

# Q3 - once we have the weights, what then? I.e. how to do step 6?

# Q4 - specific to KNN classifier
# What is a 'class' here? Is it binary? Or is there a class for every pair of faces?
# If each pair of images is a distinct 'class', then wouldn't each 'class' only contain 2 images?
# Then the probability of being in each 'class' would be tiny? How to decide 'class'? If there's only + and - class.

# Q5 - purpose of iterating through all the classes
# Moreover, what does it mean to sub together all the multiplied probabilities of each class? i.e. what is the
# point of iterating through each class c and summing? Guess answer: this is to find the total probability that
# the two image vectors belong to the same class, whatever class it is.

# Q6 - for mKnn step 5, do we decide a threshold for probability to convert from a number between 0<x<1 to
# binary classifier output?


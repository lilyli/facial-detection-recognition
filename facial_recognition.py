import os
import numpy as np
from scipy.spatial import distance

#################### Implementation of LDML Classifier #######################

# An implemention of the LDML technique in Guillaumin et. al (2009). This technique
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
    dist = distance.mahalanobis(vector0, vector1, inverse_of_covariance_mx) # how to find inverse of covariance?
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

# Function to extract feature vector
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

                d = mahalanobis_dist(base_vt, companion_vt, ???????) # inverse of covariance matrix?
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

# Predicts LDML probability, using trained weights, for a single image.
# Given an feature vector (now just Mahalanobis dist, potentially more), and trained weights, return probability
# that this distance vector belongs to a pair of images that depict the same face.
def predict_ldml_singlepair(img0, img1, trained_ws):
    features0 = extract_feature_vector(img0)
    features1 = extract_feature_vector(img1)
    d = mahalanobis_dist(features0, features1, ???) # TODO: find covariance matrix
    features = [d] # can potentially append more features, in addition to Mahalanobis distance
    return sigmoid(np.dot(features, trained_ws))

# TODO: For testing, calculate the differences from a bunch of images, and apply the predict() function
# TODO: check by inspection (?)

def resolve_gtlabel(file, base_idx, companion_idx):
    # TODO: instructions below
    # find the gtlabel0 of the image with index base_idx
    # find the gtlabel1 of the image with index companion_idx
    # if gtlabel0 == gtlabel1, then return 1
    # if gtlabel0 != gtlabel1, then return 0
    return 0



#################### Implementation of mKnn Classifier #######################

# An implementation of the mKnn classifier as per Guillaumin et al. (2009). mKNN stands for Marginalized KNN,
# which means taking the marginal probability of xi and xj belonging to the same class.
# Google Books reference: https://tinyurl.com/y9kh3b9m

# The purpose of Knn is to use a database in which the data points are separated into different classes to predict
# the classification of a new sample point.

# Step 1: Calculate the feature vectors of every image in the test set.

# Step 2: Given a new test pair of images, calculate their feature vectors xi and xj.

# Step 3: Calculate the probability that we assign xi to class c using P(yi=c|xi)=ni_c/k;
# and the probability that we assign xj to class c using P(yj=c|xj)=nj_c/k, where
# ni_c is the number of neighbors of xi of class c, and nj_c is the number of neighbors of xj of class c.
# So get ni_c and nj_c, we need to iterate through all the nearest neighbors, and tally votes.

# Step 4: Sum the joint probability of belonging to the same class c, across all classes.
# p(yi=yj|xi,xj) = SUMMATION_ALL_CLASSES( p(yi=c|xi) * p(yj=c|xj) )

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


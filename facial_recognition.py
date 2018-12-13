import numpy as np
from scipy.spatial import distance

#################### Implementation of LDML Classifier #######################

# An implemention of the LDML technique in Guillaumin et. al (2009). This technique
# learns a linear transformation by optimizing the parameters of a logistic discriminant model,
# which is used to determine if two images depict the same person or not.

# P(yi = yj | xi, xj, M, b) = (1 + exp(dM(xi, xj) - b))^-1
# where yi and yj are images, xi and xj are feature vectors of those images, dM is Mahalanobis distance, b a bias term.
# Both M and b are learned using gradient ascent on the log-likelihood of P(yi = yj | xi, xj, M, b).

# Step 1: Preparing images.
# Step 2: Extracting feature vector from an image.
# Step 3: Calculating Mahalanobis Distance for each pair of images.
# Step 4: Update weights with gradient; consider bias term (include intercept so that coefficients are unbiased, and
# that the probability of success is nonzero when other inputs are zero).
# Step 5: Run the gradient-ascent-constructed logistic regression model, to return the weights.
# Step 6: Produce final scores, and make prediction

# Function to extract feature vector
def extract_feature_vector(img):
    vector = img[0] # temporary
    return vector

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
def gradient_ascent(gt_dist, gt_labels, num_steps, learning_rate):

    # we are optimizing for gt_dist and the intercept (bias) term, so create a new features container
    intercept = np.ones((gt_dist.shape[0]), 1)
    features = np.hstack((intercept, gt_dist))

    # initialize weights
    weights = np.zeros(features.shape[1])

    for epoch in xrange(num_steps):
        scores = np.dot(features, weights) # set up, with weights/coefficients multiply by input (gt_dist)
        sigmoid_output = sigmoid(scores)

        # Update weights matrix with gradient
        output_error_signal = gt_labels - sigmoid_output
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

    return weights

# step 6: not sure??

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

# Q2 - once we have the weights, what then? I.e. how to do step 6?

# Q3 - specific to KNN classifier
# What is a 'class' here? Is it binary? Or is there a class for every pair of faces?
# If each pair of images is a distinct 'class', then wouldn't each 'class' only contain 2 images?
# Then the probability of being in each 'class' would be tiny? How to decide 'class'? If there's only + and - class.

# Q4 - purpose of iterating through all the classes
# Moreover, what does it mean to sub together all the multiplied probabilities of each class? i.e. what is the
# point of iterating through each class c and summing? Guess answer: this is to find the total probability that
# the two image vectors belong to the same class, whatever class it is.

# Q5 - for mKnn step 5, do we decide a threshold for probability to convert from a number between 0<x<1 to
# binary classifier output?


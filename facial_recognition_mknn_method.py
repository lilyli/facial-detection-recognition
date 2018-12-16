import os
import numpy as np
from scipy.spatial import distance

#################### Implementation of mKnn Classifier #######################

# An implementation of the mKnn classifier as per Guillaumin et al. (2009). mKNN stands for Marginalized KNN,
# which means taking the marginal probability of xi and xj belonging to the same class.
# Google Books reference: https://tinyurl.com/y9kh3b9m

# The purpose of Knn is to use a database in which the data points are separated into different classes to predict
# the classification of a new sample point.

'''
    Find nearest neighbors for each image

    Returns:
        all_histograms - histogram count of identity class frequencies, for every image 
'''
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

'''
    Find probability of two image pairs being in the same identity class

    Returns:
        probability - probability based on Guillaumin paper 
'''
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

# facial-detection-recognition
CMSC 25050 Final Project

Lily Li and Scott Wang

facial_detection_util.py: Code for several utility functions for facial detection. Contains:
-compute_cell_histogram(): Takes in an 8x8 cell and computes a histogram for the cell
-Other functions including conv_1d_centered(), canny_nmax(), etc.

facial_detection_training.py: Code to build and train the facial detection model

facial_detection_testing.py: Code to test the facial detection model

facial_recognition_ldml_method.py: Code for the LDML method for facial recognition

facial_recognition_mknn_method.py: Code for the MKNN method for facial recognition

final_detection_recognition_pipeline.py: Code for the full facial detection and recognition pipeline

linear_svm, linear_svc, linear_svc_1: saved facial detection models

create_facial_detection_data.py: Code to create training and testing data (cropping out faces and non-faces from labelled WIDER Faces data)

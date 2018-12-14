# facial-detection-recognition
CMSC 25050 Final Project

Lily Li and Scott Wang

facial_detection_util.py: Code for several utility functions for facial detection. Contains:
-compute_cell_histogram(): Takes in an 8x8 cell and computes a histogram for the cell
-Other functions including conv_1d_centered(), canny_nmax(), etc.

facial_detection_training.py: Code to build and train the facial detection model

create_facial_detection_data.py: Code to create training and testing data (cropping out faces and non-faces from labelled WILDER data)

final_detection_recognition_pipeline.py: Code for the full facial detection and recognition pipeline, but I don't think we'll get to this step.

facial_detection_testing.py- Ignore, I'm going to delete this later

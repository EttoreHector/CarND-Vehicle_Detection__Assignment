Vehicle Detection Project

The goals / steps of this project are the following:

* Performing a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, applying a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implementing a sliding-window technique and use a trained classifier to search for vehicles in images.
* Running the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

This submission comprises the following files:

* Writeup.pdf - A file describing how the pipeline of the project was implemented.
* README.txt - The present file.
* classifier.py - python code that creates and train a classifier (SVM) for vehicle detection.
* find_cars.py - python code containing the pipeline for vehicle detection.
* functions.py - python code containing the functions needed for the pipeline.
* model.pkl - file containing the trained classifier plus other parameters necessary for feature extraction.
* project_video_output.mp4 - video showing the bounding box for the detected cars in the project_video.mp4.
* teset_video_output.mp4 - video showing the bounding box for the detected cars in the test_progect.mp4.
* output_images - folder containing images examples from the vehicle detection pipeline.
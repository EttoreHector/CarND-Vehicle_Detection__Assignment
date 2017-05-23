# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:44:12 2017

@author: ettore
"""

import numpy as np
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from functions import extract_features

# Read in car and non-car images
cars = []
notcars = []
images_cars = glob.glob('./vehicles/KITTI_extracted/*.png')
for image in images_cars:
    cars.append(image)
images_notcars = glob.glob('./non-vehicles/Extras/*.png')
for image in images_notcars:
    notcars.append(image)

    
# Set parameters for feature extraction
spatial = (32,32) #(32,32)
histbin = 32
cspace = 'HLS'#'LUV'#''HLS'
cspaceHog = 'YCrCb'
pix_per_cell = 8
orient = 9
cell_pb = 2
hog_channel = 1

# Extract features from car and non-car images
car_features = extract_features(images_cars, cspaceHog = cspaceHog,
                                cspace=cspace, spatial_size=spatial,
                                hist_bins=histbin, hist_range=(0, 256),
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_pb, hog_channel=hog_channel)
notcar_features = extract_features(images_notcars, cspaceHog = cspaceHog, 
                                   cspace=cspace, spatial_size=spatial,
                                   hist_bins=histbin, hist_range=(0, 256),
                                   orient=orient,pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_pb, hog_channel=hog_channel)

# Visualize some picture
#imgcar = mpimg.imread(images_cars[0])
#imgnotcar = mpimg.imread(images_notcars[1])
#fig = plt.figure()
#plt.imshow(imgcar)
#plt.title('Car image')
#fig = plt.figure()
#plt.imshow(imgnotcar)
#plt.title('Non-car image')

#features, hog_imgcar0 = hog(imgcar[:,:,0], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_pb, cell_pb), visualise=True, feature_vector=False)
#features, hog_imgcar1 = hog(imgcar[:,:,1], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_pb, cell_pb), visualise=True, feature_vector=False)
#features, hog_imgcar2 = hog(imgcar[:,:,2], orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), cells_per_block=(cell_pb, cell_pb), visualise=True, feature_vector=False)

#fig = plt.figure()
#plt.imshow(hog_imgcar0, cmap='gray')
#plt.title('Hog features for car image chanel 1')
#fig = plt.figure()
#plt.imshow(hog_imgcar1, cmap='gray')
#plt.title('Hog features for car image chanel 2')
#fig = plt.figure()
#plt.imshow(hog_imgcar2, cmap='gray')
#plt.title('Hog features for car image chanel 3')


# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
                      
# Scale the features
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,
                                                    random_state=rand_state)

# Create and train model
svc = LinearSVC()
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 7
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

model = {}
model['svc'] = svc
model['scaler'] = X_scaler
model['spatial'] = spatial
model['histbins'] = histbin
model['orient'] = orient
model['cspaceHog'] = cspaceHog
model['cspace'] = cspace
model['pix_per_cell'] = pix_per_cell
model['cell_per_block'] = cell_pb 
model['hog_channel'] = hog_channel
     
     
print (model)

with open('model.pkl', 'wb') as output:
    pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

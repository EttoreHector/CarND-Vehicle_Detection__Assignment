# -*- coding: utf-8 -*-
"""
Created on Wed May 17 08:51:28 2017

@author: ettore
"""
import glob
import matplotlib.image as mpimg
from functions import transform_image
import scipy.misc

white_cars = []
images_cars = glob.glob('./vehicles/KITTI_extracted/WhiteCar*.png')
for image in images_cars:
    white_cars.append(image)
    
i=1
for img_path in white_cars:
    image = mpimg.imread(img_path)
    for j in range(20):
        imgTemp = transform_image(image,5,5,4)
        scipy.misc.imsave('./vehicles/KITTI_extracted/www'+str(i)+'.png', imgTemp)
        i += 1
#        fig1 = plt.figure()
#        plt.imshow(imgTemp)
#    break
        

other_notcars = []
images_notcars = glob.glob('./non-vehicles/Extras/Other*.png')
for image in images_notcars:
    other_notcars.append(image)
    
i=1
for img_path in other_notcars:
    image = mpimg.imread(img_path)
    for j in range(20):
        imgTemp = transform_image(image,5,5,4)
        scipy.misc.imsave('./non-vehicles/Extras/rrr'+str(i)+'.png', imgTemp)
        i += 1
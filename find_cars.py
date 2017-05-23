# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:39:54 2017

@author: ettore
"""

from functions import (find_cars, add_heat, apply_threshold, 
                       draw_labeled_bboxes, overimpose_heat_maps,
                       subtract_heat_maps)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import numpy as np
import cv2
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

def track_cars_on_video(img):
    
    global boxes
    global frame
    global nframes
    global nbox_in_frame
    global heat_tot
    
    #Extrackt model and features parameters from pickle file
    dist_pickle = pickle.load(open('model.pkl', 'rb'))
    svc = dist_pickle['svc']
    X_scaler = dist_pickle['scaler']
    spatial = dist_pickle['spatial']
    histbins = dist_pickle['histbins']
    orient = dist_pickle['orient']
    cspaceHog = dist_pickle['cspaceHog']
    cspace = dist_pickle['cspace']
    pix_per_cell = dist_pickle['pix_per_cell']
    cell_per_block = dist_pickle['cell_per_block']
    hog_channel = dist_pickle['hog_channel']
    
    #img = mpimg.imread('./test_images/test1.jpg')
    
    #print('cspace = ', cspace)
    #print('cspaceHog = ', cspaceHog)
    
    ystart = 400
    ystop = 500#450
    scale = 0.8
    boxes1, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial, histbins, 
                        cspace = cspace, cspaceHog = cspaceHog, hog_channel = hog_channel)
    
    ystart = 350
    ystop = 550 # 650
    scale = 1.2
    boxes2, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial, histbins, 
                        cspace = cspace, cspaceHog = cspaceHog, hog_channel = hog_channel)
    
    ystart = 350
    ystop = 600
    scale = 1.6
    boxes3, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial, histbins, 
                        cspace = cspace, cspaceHog = cspaceHog, hog_channel = hog_channel)
    
    ystart = 300
    ystop = 650
    scale = 2.0
    boxes4, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial, histbins, 
                        cspace = cspace, cspaceHog = cspaceHog, hog_channel = hog_channel)
    
    ystart = 300
    ystop = 700
    scale = 2.5
    boxes5, out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, 
                        pix_per_cell, cell_per_block, spatial, histbins, 
                        cspace = cspace, cspaceHog = cspaceHog, hog_channel = hog_channel)
    
    

    
    boxes_ALL = boxes5 + boxes4 + boxes3 + boxes2 + boxes1
    
    draw_img = np.copy(img)
    #for box in boxes_ALL:
    #    cv2.rectangle(draw_img,(box[0][0], box[0][1]),(box[1][0],box[1][1]),(0,0,255),6) 
    
#    fig1 = plt.figure()
#    plt.imshow(img)
#    plt.title('Original image')
#    
#    fig1 = plt.figure()
#    plt.imshow(draw_img)
#    plt.title('Car detection using sliding window')
    

    # Create heat-map for current frame
    #heat = np.zeros_like(img[:,:,0]).astype(np.float)
    #heat = add_heat(heat,boxes_ALL)
    # Apply threshold to remove single hit on current frame
    #heat = apply_threshold(heat,1)
    # Overimpose with previous heat-maps
    #heat_tot = overimpose_heat_maps(heat_tot, heat)    
    
    # Keep track of the number of boxes in last nframes frames
    n_boxes = len(boxes_ALL)  
    nbox_in_frame.append(n_boxes) 
    
    if frame > nframes:
        #box_to_be_subtracted = boxes[:nbox_in_frame[0]]
        #heat_tot = subtract_heat_maps(heat_tot, box_to_be_subtracted)
        del boxes[:nbox_in_frame[0]]
        nbox_in_frame.pop(0)
    boxes = boxes_ALL + boxes
    frame += 1
    
    # Create empty heat-map
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,boxes)        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,4) # 7 for test video
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # Draw singleboxes around cars
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
#    fig2 = plt.figure()
#    #plt.subplot(121)
#    plt.imshow(draw_img)
#    plt.title('Car Positions')
#    fig3 = plt.figure()
#    #plt.subplot(122)
#    plt.imshow(heatmap, cmap='hot')
#    plt.title('Heat Map')
#    #fig.tight_layout()
    
    return draw_img

boxes = []
nframes = 2 # 5 for test video
frame = 1
nbox_in_frame = []

img = mpimg.imread('./test_images/test1.jpg')
heat_tot = np.zeros_like(img[:,:,0]).astype(np.float)

#track_cars_on_video()

video_output = 'test_video_output_2_4.mp4'
clip = VideoFileClip('./test_video.mp4')
new_clip = clip.fl_image(track_cars_on_video)
new_clip.write_videofile(video_output)

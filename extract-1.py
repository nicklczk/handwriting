# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 19:41:07 2018

@author: Sean Rice
"""
import sys
import numpy as np
import os
import cv2
from sklearn import svm
import imutils
import matplotlib.pyplot as plt
import math

def get_image_to_show(im):
    '''
    Return an image that is ready for display as a uint8 in the range 0..255
    '''
    min_g = np.min(im)
    max_g = np.max(im)
    if min_g >= 0 and max_g <= 255 and im.dtype == np.uint8:
        im_to_show = im
    else:
        im_to_show = (im-min_g) / (max_g - min_g) * 255
        im_to_show = im_to_show.astype(np.uint8)
    return im_to_show

def plot_pics(image_list, num_in_col=2, title_list=[]):
    '''
    Given a list of images, plot them in a grid using PyPlot
    '''
    if len(image_list) == 0:
        return
    
    if len(image_list[0].shape) == 2:
        plt.gray()
        
    num_rows = math.ceil(len(image_list)/num_in_col)
    if num_in_col > 2 and len(image_list) > 2:
        plt.figure(figsize=(12,12))
    else:
        plt.figure(figsize=(15,15))

    for i in range(len(image_list)):
        im = image_list[i]
        plt.subplot(num_rows, num_in_col, i+1)

        im_to_show = get_image_to_show(im)
        plt.imshow(im_to_show)
        if i < len(title_list):
            plt.title(title_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    
def extract_regions(img_in):
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = imutils.resize(img_in, width=300)
    
    wy = img.shape[0]/img_in.shape[0]
    wx = img.shape[1]/img_in.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0,
    	ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    
    print("Energy")
    plt.imshow(gradX, cmap="gray")
    plt.show()
    
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    
    gradX = cv2.GaussianBlur(gradX,(5,5),0)
    thresh = cv2.threshold(gradX, 0, 255,
    	cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
     
    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    
    
    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []
    
    # loop over the contours
    for (i, c) in enumerate(cnts):
    	# compute the bounding box of the contour, then use the
    	# bounding box coordinates to derive the aspect ratio
    	(x, y, w, h) = cv2.boundingRect(c)
    	ar = w / float(h);locs.append((int(x//wx), int(y//wy), int(w//wx), int(h//wy)))
        
    plt.imshow(thresh, cmap="gray")
    plt.show()
        
    return locs

def extract_letters(img):
    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
#    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#    kernel = np.ones((3,3),np.uint8)
#    thresh = cv2.erode(thresh, rectKernel)
    
    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []
    
    # loop over the contours
    for (i, c) in enumerate(cnts):
    	# compute the bounding box of the contour, then use the
    	# bounding box coordinates to derive the aspect ratio
    	(x, y, w, h) = cv2.boundingRect(c)
    	ar = w / float(h);locs.append((x, y, w, h))
    
    im_box = img.copy()
    for (x,y,w,h) in locs:
        cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0))
        
#    plt.imshow(im_box[:,:,::-1], cmap="gray")
#    plt.show()
    
    return locs

   
if __name__ == "__main__":
    image = cv2.imread("test.PNG")
    regions = extract_regions(image)
    
    im_box = image.copy()
    for (x,y,w,h) in regions:
        cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0))
#        im_sm = image[y:y+h, x:x+w]
#        reg = extract_letters(im_sm)
    plt.imshow(im_box[:,:,::-1], cmap="gray")
    plt.show()
    
    regions = extract_letters(image)
    
    im_box = image.copy()
    for (x,y,w,h) in regions:
        cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0))
#        im_sm = image[y:y+h, x:x+w]
#        reg = extract_letters(im_sm)
    plt.imshow(im_box[:,:,::-1], cmap="gray")
    plt.show()
    
    
    
#        im_sm = image[y:y+h, x:x+w]
#        regions = extract_letters(im_sm)
#out = image;
#    
#    #for (x,y,w,h) in locs:
#    #    cv2.rectangle(out, (x,y), (w+x,h+y), (255,0,0))
#        
#    #plt.imshow(out[:,:,::-1], cmap="gray")
#    #plt.show()
#    
#sub_imgs = [image[y:y+h, x:x+w] for (x,y,w,h) in locs]
#plot_pics(sub_imgs[:16], num_in_col=4)

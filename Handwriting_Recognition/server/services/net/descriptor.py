# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:35:39 2018

@author: Sean Rice
"""

import sys
import numpy as np
import os
import cv2
import math
import scipy
import matplotlib.pyplot as plt


def norm_range(x, hi):
    return hi * (x - np.min(x)) / (np.max(x)-np.min(x))

''' This implementation comes from hw6.pdf  '''
def color_descriptor(img):
    # Generate a small image of random values.
    Y, X = img.shape[0:2]
    t = 4
    b = 4

    desc = np.zeros_like((0))

    for i in range(b):
        for j in range(b):
            lo = (i*Y//b, j*X//b)
            hi = ((i+1)*Y//b, (j+1)*X//b)

            im_sq = img[lo[0]:hi[0], lo[1]:hi[1]]
            pixels = im_sq.reshape(-1, 3)
            hist, _ = np.histogramdd(pixels, (t, t, t))
            if (i == 0 and j == 0):
                desc = hist.flatten()
            else:
                desc = np.concatenate((desc, hist.flatten()))
#    print("histogram shape:", hist.shape) # should be t, t, t
#    print("histogram:\n", hist)
#    print(np.sum(hist)) # should sum to M*N
    return norm_range(desc)

def basic_descriptor(img):
    img_r = cv2.resize(img, dsize=(24,32))
    return img_r.flatten()

def binary_descriptor(img):
    img_r = cv2.resize(img, dsize=(28,28))
    if (img.shape == (28,28,3)):
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_r = np.where(img_r > 127, 1, 0)
    return img_r.flatten()

def binary_descriptor_inv(img):
    img_r = cv2.resize(img, dsize=(28,28))
    if (img.shape == (28,28,3)):
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_r = np.where(img_r > 127, 0, 1)

    return img_r.flatten()

def getBestShift(img):
    cy,cx = scipy.ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def format_img_emnist(img, black_on_white=True, desc_method=binary_descriptor):

    img_r = cv2.resize(img, dsize=(28,28))
    if (black_on_white==True):
        gray = cv2.cvtColor(255 - img_r, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    # plt.imshow(gray)
    # plt.show()
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = 0,0
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    return desc_method(gray)

'''  Load batches of images into memory. For each image we:
        Calculate descriptor vector
        Write to file  '''
def write_descriptors(file_names, out_filename, img_batch_size = 20, desc_method=basic_descriptor):
    out_file = open(out_filename, "w")

    imgs = []
    img_names = []
    for i in range(0, len(file_names), img_batch_size):
        # On last batch
        if (i + img_batch_size ) >= len(file_names):
            img_names = file_names[i:]
        else:
            img_names = file_names[i:i+20]

        print("Batch", i//20)

        # Grab image batch
        imgs = imread_list(img_names)

        # Calculate descriptor vectors of image batch
        imgs_desc = [color_descriptor(img) for img in imgs]

        # Write descriptors to file
        for j in range(0, len(imgs_desc)):
            line = ' '.join(map(str, imgs_desc[j].tolist()))
            out_file.write(line + "\n")

    out_file.close()


def img_names_path(dir_name):
    img_list = os.listdir(dir_name)
    img_list.sort()
    img_list = [(dir_name + "/" + name) for name in img_list if ".jpeg" in name.lower()]

    return img_list

def imread_list(img_list):
    imgs = [cv2.imread(name) for name in img_list if os.path.isfile(name)]
    #os.chdir(os.path.dirname(sys.argv[0]))
    return imgs

if __name__ == "__main__":
    commands = ["-d", "-f"]

    filePaths = []
    dirPaths = []

    imgs = []
    img_names = []

    out_dir = ""

    '''  Parse program arguments
         I would either leave these or remove them. '''
    for i in range(0, len(sys.argv)):
        arg = sys.argv[i]

        # Get output-dir
        if (arg == "-o"):
            out_dir = sys.argv[i+1]

        # Parse a list of files separated by spaces
        if (arg == "-f"):
            for j in range(i+1, len(sys.argv)):
                filePath = sys.argv[j]

                if filePath in commands:
                    break;

                filePaths.append(filePath)

        # Parse a list of directories separated by space
        if (arg == "-d"):
            for j in range(i+1, len(sys.argv)):
                dirPath = sys.argv[j]

                if dirPath in commands:
                    break;

                dirPaths.append(dirPath)


    '''  Grab all file paths from the directories  '''
    for dirPath in dirPaths:
        filePaths.extend(img_names_path(dirPath))

    '''  Write all descriptors to a file  '''
    write_descriptors(filePaths, out_dir, desc_method=basic_descriptor)

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 16:35:39 2018

@author: Sean Rice
"""

import sys
import numpy as np
import os
import cv2


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
    
'''  Load batches of images into memory. For each image we:
        Calculate descriptor vector
        Write to file  '''
def write_color_descriptors(file_names, out_filename, img_batch_size = 20, desc_method=color_descriptor):
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
    
    '''  Parse program arguments  '''
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
    write_color_descriptors(filePaths, out_dir)
                
            
    

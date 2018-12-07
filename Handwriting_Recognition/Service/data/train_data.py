'''
train_data.py

Sean Rice, 12/7/2018

Create training data from a directory of filesself.

Run: train_data.py <parent_dir>

The parent directory should be in the following format:
    parent
    |----class0
    |    |----im1.ext
    |    |----im2.ext
    |    |----  ...
    |
    |----class0
    |    |----im1.ext
    |    |----im2.ext
    |    |----  ...

Each class
Parse images into descriptor vectors.

Write output data is in the following format:
    {
        "classname1" : {
            "im1" : [1, 2, ...],
            "im2" : [3, 4, ...],
            ...
        },
        "classname2" : {
            "im1" : [1, 2, ...],
            "im2" : [3, 4, ...],
            ...
        },
        ...
    }
'''

import sys
import json
import os
import cv2
import descriptor
import numpy

'''
Parse json string from file into python dict
'''
def read_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

'''
Write json string to file from python dict
'''
def write_json(filename, data={}):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile, indent=4)

'''
Create a list containing all sub-directories in the input path
'''
def list_subdirs(dir_name):
    img_list = os.listdir(dir_name)
    img_list.sort()
    img_list = [(dir_name + "/" + name) for name in img_list if os.path.isdir(dir_name + "/" + name)]

    return img_list

def list_images(dir_name):
    img_list = os.listdir(dir_name)
    img_list.sort()
    img_list = [(dir_name + "/" + name) for name in img_list if os.path.splitext(name.lower())[1] in [".jpeg", ".jpg", ".png" ]]

    return img_list

def create_data(child_dirs) :
    data = {}

    # For each child dir, make class training data
    for subdir in child_dirs[:2]:
        classname = os.path.basename(subdir)
        print(classname)

        img_names = list_images(subdir)
        imgs = imread_list(img_names)

        # Parse descriptors
        descs = {}
        for i in range(len(imgs)):
            desc = descriptor.basic_descriptor(imgs[i]).tolist()
            descs[img_names[i]] = {
                "desc" : desc,
                "len" : len(desc)
            }
        data[classname] = descs
    return data

def create_train_data(data):
    X = []
    Y = []
    for cl in data:
        for im_name, desc in data[cl].items():
            X.append(desc["desc"])
            Y.append(cl)
    train_data = {
        "X" : X,
        "Y" : Y
    }
    return train_data


'''
Read a list of images using cv2
'''
def imread_list(img_list):
    imgs = [cv2.imread(name) for name in img_list if os.path.isfile(name)]
    #os.chdir(os.path.dirname(sys.argv[0]))
    return imgs

if __name__ == "__main__":
    # Parse Arguments to get parent dir
    parent_dir = sys.argv[1]

    # Get child dirs
    child_dirs = list_subdirs(parent_dir)

    data = create_data(child_dirs)
    train_data = create_train_data(data)

    write_json("data.json", data)
    write_json("train_data.json", train_data)

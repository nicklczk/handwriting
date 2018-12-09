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
        json.dump(data, outfile)

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
    for subdir in child_dirs:
        classname = os.path.basename(subdir)
        print(classname)

        img_names = list_images(subdir)
        imgs = imread_list(img_names)

        # Parse descriptors
        descs = {}
        for i in range(len(imgs)):
            desc = descriptor.binary_descriptor_inv(imgs[i]).tolist()
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

def create_data_ttv(child_dirs):
    data = {
        "train" : {
            "X" : [],
            "Y" : []
        },
        "test" :  {
            "X" : [],
            "Y" : []
        },
        "valid" : {
            "X" : [],
            "Y" : []
        }
    }

    # For each child dir, make class training data
    for subdir in child_dirs:
        classname = os.path.basename(subdir)
        print(classname)

        img_names = list_images(subdir)
        imgs = imread_list(img_names)

        l = len(imgs)
        i_1 = int(l*.70)
        i_2 = int(l*.85)

        for i in range(0, i_1):
            desc = descriptor.binary_descriptor_inv(imgs[i]).tolist()
            data["train"]["X"].append(desc)
            data["train"]["Y"].append(classname)
        for i in range(i_1, i_2):
            desc = descriptor.binary_descriptor_inv(imgs[i]).tolist()
            data["test"]["X"].append(desc)
            data["test"]["Y"].append(classname)
        for i in range(i_2, l):
            desc = descriptor.binary_descriptor_inv(imgs[i]).tolist()
            data["valid"]["X"].append(desc)
            data["valid"]["Y"].append(classname)

    return data


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

    print("Creating data...")
    data = create_data_ttv(child_dirs)

    # write_json("data.json", data)
    write_json(sys.argv[2], data)

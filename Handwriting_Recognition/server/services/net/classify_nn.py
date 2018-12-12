# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 01:13:09 2018

@author: Sean Rice
"""

import sys
import json
import numpy as np
import os
import cv2
from sklearn import svm, metrics
# import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import argparse
import torch.nn as nn
import torch.nn.functional as F
from . import extract1
from . import descriptor

'''
Load the parameters of a NN from filename

Model Properties:
    Name
    Training data
    Testing data
    Learning rate
Weights and Biases
'''

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

class Net(nn.Module):

    def __init__(self, N=[1000, 100, 10, 1]):
        super(Net, self).__init__()

        # Create three fully connected layers, two of which are hidden and the third is
        # the output layer.  In each case, the first argument o Linear is the number
        # of input values from the previous layer, and the second argument is the number
        # of nodes in this layer.  The call to the Linear initializer creates a PyTorch
        # functional that in turn adds a weight matrix and a bias vector to the list of
        # (learnable) parameters stored with each Net object.  These weight matrices
        # and bias vectors are implicitly initialized using a normal distribution
        # with mean 0 and variance 1
        self.fc1 = nn.Linear(N[0], N[1], bias=True)
        self.fc2 = nn.Linear(N[1], N[2], bias=True)
        self.fc3 = nn.Linear(N[2], N[3], bias=True)

    def forward(self, x):
        #  The forward method takes an input Variable and creates a chain of Variables
        #  from the layers of the network defined in the initializer. The F.relu is
        #  a functional implementing the Rectified Linear activation function.
        #  Notice that the output layer does not include the activation function.
        #  As we will see, that is combined into the criterion for the loss function.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NN_Classifier:
    def __init__(self, name=None, filename=None):
        self.name = name
        self.savepath = filename

        self.curr_epoch = 0
        self.learn_rate = 0.1

        self.X_train = None
        self.Y_train = None

    def save_path(self, path):
        self.savepath = path
        self.name = os.path.splitext(os.path.basename(path))[0]
        print(self.name)

    def train_data(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def init_net(self, len_nodes=[1000,100,10,1], learn_rate=.01) :
        self.nodes = len_nodes
        self.learn_rate = learn_rate
        self.model = Net(len_nodes).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate)

    def load(self, filename=None):
        path = None

        # Use the filename if its give, otherwise fallback to self.savepath
        if (filename != None):
            path = filename
        else:
            path = self.savepath

        checkpoint = torch.load(path)
        self.nodes = checkpoint["nodes"]
        self.curr_epoch = checkpoint["epoch"]
        self.learn_rate = checkpoint["learn_rate"]
        self.init_net(self.nodes, self.learn_rate)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.train()

    def save(self, filename=None):
        path = None

        # Use the filename if its give, otherwise fallback to self.savepath
        if (filename != None):
            path = filename
        else:
            path = self.savepath

        torch.save({
                        'nodes': self.nodes,
                        'epoch': self.curr_epoch,
                        'learn_rate': self.learn_rate,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, path)

        print("Model written to", path)

    def train(self, batch_size=32):
        print(self.learn_rate)

        n_train = X_train.shape[0]
        n_batches = int(np.ceil(n_train / batch_size))

        print("Starting epoch",self.curr_epoch+1)
        #  Create a random permutation of the indices of the row vectors.
        indices = torch.randperm(n_train)

        #  Run through each mini-batch
        for b in range(n_batches):

            #  Use slicing (of the pytorch Variable) to extract the
            #  indices and then the data i/nstances for the next mini-batch
            batch_indices = indices[b*batch_size: (b+1)*batch_size]
            batch_X = X_train[batch_indices].cuda()
            batch_Y = Y_train[batch_indices].cuda()

            #  Run the network on each data instance in the minibatch
            #  and then compute the object function value
            pred_Y = self.model(batch_X)

            loss = self.criterion(pred_Y, batch_Y)

            #  Back-propagate the gradient through the network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.curr_epoch += 1

    def model_accuracy(self, X, Y, batch_size=32):
        n_train = X.shape[0]

        n_batches = int(np.ceil(n_train / batch_size))

        #  Create a random permutation of the indices of the row vectors.
        indices = torch.randperm(n_train)

        acc = 0
        num_eq = 0
        num_diff = 0
        for b in range(n_batches):
            #  Use slicing (of the pytorch Variable) to extract the
            #  indices and then the data instances for the next mini-batch
            batch_indices = indices[b*batch_size: (b+1)*batch_size]
            batch_X = X[batch_indices].cuda()
            batch_Y = Y[batch_indices].cuda()

            #  Run the network on each data instance in the minibatch
            #  and then compute the object function value
            pred_Y = self.model(batch_X)
            a, e, d = success_rate(pred_Y, batch_Y)
            num_eq += e
            num_diff += d
        acc = num_eq / float(num_eq + num_diff)
        # print('Training success rate:', acc)
        return acc

    def confusion_matrix(self, X, Y):
        pred_Y = torch.max(self.model(X.cuda()), 1)[1].detach().cpu().numpy()
        conf_matrix = metrics.confusion_matrix(pred_Y, Y)
        return conf_matrix


def success_rate(pred_Y, Y):
    _,pred_Y_index = torch.max(pred_Y, 1)
    num_equal = torch.sum(pred_Y_index.data == Y.data).item()
    num_different = torch.sum(pred_Y_index.data != Y.data).item()
    rate = num_equal / float(num_equal + num_different)
    return rate, num_equal, num_different # rate.item()

def read_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

def load_data(filename):
    print("Reading training images...", end="", flush=True)
    data = read_json(filename)
    print("\tDone!\n")

    print("Creating data...")
    X_train = [[int(i) for i in d] for d in data["train"]["X"]]
    X_test  = [[int(i) for i in d] for d in data["test" ]["X"]]
    X_valid = [[int(i) for i in d] for d in data["valid"]["X"]]

    X_train = torch.Tensor(np.array(X_train).astype(np.uint8))
    X_test  = torch.Tensor(np.array(X_test).astype(np.uint8))
    X_valid = torch.Tensor(np.array(X_valid).astype(np.uint8))

    c2l = []    # Class number to label
    l2c = {}    # Label to class number

    for cl in data["train"]["Y"]:
        if cl not in c2l:
            c2l.append(cl)
            l2c[cl] = len(c2l)

    Y_train = [l2c[d] for d in data["train"]["Y"]]
    Y_test  = [l2c[d] for d in data["test"]["Y"]]
    Y_valid = [l2c[d] for d in data["valid"]["Y"]]

    Y_train = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_train))))
    Y_test = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_test))))
    Y_valid = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_valid))))

    return (X_train, Y_train), (X_test, Y_test), (X_valid, Y_valid)

# def create_new(filename, learn_rate)

'''
TTV short for Train, Test, Validation
'''
def split_data_ttv(X):
    tr = 0.15
    te = 0.75
    va = 0.15

    len_data = X.shape[0]

    indices = torch.randperm(len_data)
    X1 = X[indices]

    i_1 = int(len_data*tr)
    i_2 = int(len_data*(tr+te))

    X_tr = X1[0:i_1]
    X_te = X1[i_1:i_2]
    X_va = X1[i_2:]

    return X_tr, X_te, X_va

def classify(cl_filepath, im_filepath, letter_rects):
    classifier = NN_Classifier()

    # Load an existing classifier from file
    if (cl_filepath!= None and os.path.isfile(cl_filepath)):
        classifier.save_path(cl_filepath)
        classifier.load()

        print("epoch", classifier.curr_epoch)

        # Load image from files
        img = cv2.imread(im_filepath)

        predictions = []
        im_out = img.copy()
        for rect in letter_rects:
            x,y,w,h = rect
            subim = img[y-20:y+h+20, x-20:x+w+20]


            desc = descriptor.binary_descriptor_inv(subim)

            X = torch.Tensor(np.array(desc).astype(np.uint8).reshape(-1,desc.shape[0]))

            pred_Y = classifier.model(X.cuda())

            class_ind = torch.max(pred_Y, 1)[1].detach().cpu().numpy()

            pred = chr(class_ind[0]+64) if class_ind[0] > 0 and class_ind[0] < 27 else '?'
            predictions.append(pred)

            print(pred, class_ind[0])
            # cv2.imshow("subim", subim)
            # cv2.waitKey(0)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im_out,pred,(x,y), font, 2,(0,0,255),2,cv2.LINE_AA)
            cv2.rectangle(im_out, (x, y), (x+w, y+h), (255, 0, 0))
            cv2.imwrite("images/out.jpg", im_out)

        return predictions



#classifier.save("net_test.pt")
if __name__ == "__main__":
    # Define arguments
    cli = argparse.ArgumentParser()
    cli.add_argument("--new_classifier", type=str)
    cli.add_argument("--load_classifier", type=str)
    cli.add_argument("--train", type=str)
    cli.add_argument("--test", type=str)
    cli.add_argument("--batch_size", type=float, default=512)
    cli.add_argument("--epochs", type=int, default=10)
    cli.add_argument("--lr", type=float)
    cli.add_argument("--stats", type=str)

    # Parse args string into args object
    args = cli.parse_args()


    train_dirs = args.train
    test_dirs = args.test

    train, test, valid = load_data(train_dirs)

    X_train, Y_train = train
    X_test, Y_test = test
    X_valid, Y_valid = valid

    #
    # X_test = torch.Tensor(np.vstack(X_2).astype(np.uint8))
    # Y_test = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_2))))
    # params[5].size()) # ... the layer 5 bias vector
    #

    classifier = NN_Classifier()

    # Create a new classifier
    if (args.new_classifier != None):
        lr = args.lr if args.lr != None else 0.1
        classifier.init_net(len_nodes=[X_train.shape[1],250,100,26], learn_rate=lr)
        classifier.save_path(args.new_classifier)
        classifier.save()
    # Load an existing classifier from file
    elif (args.load_classifier != None and os.path.isfile(args.load_classifier)):
        classifier.save_path(args.load_classifier)
        classifier.load()
        if args.lr != None:
            classifier.learn_rate = args.lr

    stats = {}
    if (args.stats != None):
        if (os.path.isfile(args.stats)):
            data = read_json(args.stats)
            if ("name" in data and data["name"]==classifier.name and "epoch" in data and data["epoch"]==classifier.curr_epoch):
                stats = data
            else:
                stats = {
                    "name": classifier.name,
                    "epoch": classifier.curr_epoch,
                    "test_acc": [],
                    "train_acc": [],
                    "valid_acc": [],
                    "confusion_matrix": []
                }
        else:
            stats = {
                "name": classifier.name,
                "epoch": classifier.curr_epoch,
                "test_acc": [],
                "train_acc": [],
                "valid_acc": [],
                "confusion_matrix": []
            }
    '''  Training  '''
    for i in range(100):
        classifier.train(batch_size=512)
        classifier.save()

        tr_a = classifier.model_accuracy(X_train, Y_train, batch_size=512)*100
        if ("train_acc" in stats):
            stats["train_acc"].append(tr_a)
        print("Train Accuracy: %3.1f" % tr_a)

        va_a = classifier.model_accuracy(X_valid, Y_valid, batch_size=512)
        if ("valid_acc" in stats):
            stats["valid_acc"].append(va_a)
        print("Validation Accuracy: %3.1f" % va_a)

        if ("confusion_matrix" in stats):
            stats["confusion_matrix"] = classifier.confusion_matrix(X_test, Y_test).tolist()

        if ("epoch" in stats):
            stats["epoch"] = classifier.curr_epoch

        # Write stats to file
        if (args.stats != None):
            write_json(args.stats, stats)
            print("Stats written to", args.stats)



    # plt.matshow(classifier.confusion_matrix(X_test, Y_test))
    # plt.show()
    print("Test Accuracy:", classifier.model_accuracy(X_test, Y_test))

    # plt.plot(train_acc, color="green")
    # plt.plot(test_acc, color="red")
    # plt.show()

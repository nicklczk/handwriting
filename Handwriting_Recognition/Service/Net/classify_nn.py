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
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import argparse
import torch.nn as nn
import torch.nn.functional as F

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
    print("Reading training images...")
    data = read_json(filename)

    print("Creating data...")
    X = [[int(i) for i in d] for d in data["X"]]

    c2l = []
    l2c = {}
    for cl in data["Y"]:
        if cl not in c2l:
            c2l.append(cl)
            l2c[cl] = len(c2l)

    Y = [l2c[d] for d in data["Y"]]

    return X, Y

cli = argparse.ArgumentParser()
cli.add_argument("--train", type=str)
cli.add_argument("--test", type=str)

args = cli.parse_args()

print("Training directories", args.train)
print("Testing directories", args.test)

train_dirs = args.train
test_dirs = args.test

X_1, Y_1 = load_data(train_dirs)
X_2, Y_2 = load_data(test_dirs)

X_train = torch.Tensor(np.array(X_1).astype(np.uint8))
Y_train = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_1))))

X_test = torch.Tensor(np.vstack(X_2).astype(np.uint8))
Y_test = Variable(torch.Tensor.long(torch.Tensor(np.array(Y_2))))

n_train, n_test = X_train.shape[0], X_test.shape[0]
#n_train, n_valid, n_test = 100, 100, 100

'''  Create network  '''
N0 = X_train.shape[1]
N1 = 250
N2 = 100
Nout = 52

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Create three fully connected layers, two of which are hidden and the third is
        # the output layer.  In each case, the first argument o Linear is the number
        # of input values from the previous layer, and the second argument is the number
        # of nodes in this layer.  The call to the Linear initializer creates a PyTorch
        # functional that in turn adds a weight matrix and a bias vector to the list of
        # (learnable) parameters stored with each Net object.  These weight matrices
        # and bias vectors are implicitly initialized using a normal distribution
        # with mean 0 and variance 1
        self.fc1 = nn.Linear(N0, N1, bias=True)
        self.fc2 = nn.Linear(N1, N2, bias=True)
        self.fc3 = nn.Linear(N2, Nout, bias=True)

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

#  Create an instance of this network.
net = Net().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=.0001)

#  Print a summary of the network.  Notice that this only shows the layers
print(net)

#  Write the network weights to a file
params = list(net.parameters())
# f = open("nn_weights.txt", "w+")
# for p in params:
#     f.write(p.size())
print(params[0].size()) # The parameter holding the layer 1 weight matrix
print(params[1].size()) # ... the layer 1 bias vector
print(params[2].size()) # ... the layer 2 weight matrix
print(params[3].size()) # ... the layer 3 bias vector
print(params[4].size()) # ... the layer 4 weight vector
print(params[5].size()) # ... the layer 5 bias vector

'''  Training  '''
#  Set parameters to control the process
epochs = 1000
batch_size = 64
n_batches = int(np.ceil(n_train / batch_size))

# Store Test/Training accuracy per epoch
train_acc = []
test_acc = []

for ep in range(epochs):
    print("Starting epoch",ep+1)
    #  Create a random permutation of the indices of the row vectors.
    indices = torch.randperm(n_train)

    #  Run through each mini-batch
    for b in range(n_batches):
        #  Use slicing (of the pytorch Variable) to extract the
        #  indices and then the data instances for the next mini-batch
        batch_indices = indices[b*batch_size: (b+1)*batch_size]
        batch_X = X_train[batch_indices].cuda()
        batch_Y = Y_train[batch_indices].cuda()

        #  Run the network on each data instance in the minibatch
        #  and then compute the object function value
        pred_Y = net(batch_X)

        loss = criterion(pred_Y, batch_Y)

        #  Back-propagate the gradient through the network using the
        #  implicitly defined backward function, but zero out the
        #  gradient first.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    num_eq = 0
    num_diff = 0
    for b in range(n_batches):
        #  Use slicing (of the pytorch Variable) to extract the
        #  indices and then the data instances for the next mini-batch
        batch_indices = indices[b*batch_size: (b+1)*batch_size]
        batch_X = X_train[batch_indices].cuda()
        batch_Y = Y_train[batch_indices].cuda()

        #  Run the network on each data instance in the minibatch
        #  and then compute the object function value
        pred_Y = net(batch_X)
        a, e, d = success_rate(pred_Y, batch_Y)
        num_eq += e
        num_diff += d
    acc = num_eq / float(num_eq + num_diff)
    print('Training success rate:', acc)
    train_acc.append(acc)

    indices = torch.randperm(n_test)
    for b in range(int(np.ceil(n_test / batch_size))):
        #  Use slicing (of the pytorch Variable) to extract the
        #  indices and then the data instances for the next mini-batch
        batch_indices = indices[b*batch_size: (b+1)*batch_size]
        batch_X = X_test[batch_indices].cuda()
        batch_Y = Y_test[batch_indices].cuda()

        #  Run the network on each data instance in the minibatch
        #  and then compute the object function value
        pred_Y = net(batch_X)
        a, e, d = success_rate(pred_Y, batch_Y)
        num_eq += e
        num_diff += d

    #  Compute and print the training and test loss
    acc = num_eq / float(num_eq + num_diff)
    print('Test success rate:', acc)
    test_acc.append(acc)


pred_Y = torch.max(net(X_test.cuda()),1)[1].detach().cpu().numpy()
print(pred_Y)
print(Y_test)

conf_matrix = metrics.confusion_matrix(pred_Y, Y_test)
for l in conf_matrix.tolist():
    print(l)

plt.plot(train_acc, color="green")
plt.plot(test_acc, color="red")
plt.show()

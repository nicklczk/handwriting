# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 01:13:09 2018

@author: Sean Rice
"""
import sys
import numpy as np
import os
import cv2
from sklearn import svm
import matplotlib.pyplot as plt
from . import torch
from torch.autograd import Variable
import argparse

#X1 = desc.read_desc("")
def train(feat, label, m):
    m.fit(feat, label)

def test(feat, label, m):
    sc = m.score(feat, label)
    return sc

def read_descriptors(filename):
    file = open(filename)
    X = []
    for line in file:
        line = line.replace("\n", "").split()
        X.append(line)
    return X

def get_all_images_dir(dir_name):
    img_list = os.listdir(dir_name)
    img_list.sort(reverse=True)
    img_list = [name for name in img_list if "jpeg" in name.lower()]

    imgs = [cv2.imread(dir_name + "/" + name).reshape((-1,)) for name in img_list]
    #os.chdir(os.path.dirname(sys.argv[0]))
    return imgs, img_list

def success_rate(pred_Y, Y):
    #print(pred_Y.data)
    #print(Y.data)
    '''
    Calculate and return the success rate from the predicted output Y and the
    expected output.  There are several issues to deal with.  First, the pred_Y
    is non-binary, so the classification decision requires finding which column
    index in each row of the prediction has the maximum value.  This is achieved
    by using the torch.max() method, which returns both the maximum value and the
    index of the maximum value; we want the latter.  We do this along the column,
    which we indicate with the parameter 1.  Second, the once we have a 1-d vector
    giving the index of the maximum for each of the predicted and target, we just
    need to compare and count to get the number that are different.  We could do
    using the Variable objects themselve, but it is easier syntactically to do this
    using the .data Tensors for obscure PyTorch reasons.
    '''
    _,pred_Y_index = torch.max(pred_Y, 1)
    num_equal = torch.sum(pred_Y_index.data == Y.data).item()
    num_different = torch.sum(pred_Y_index.data != Y.data).item()
    rate = num_equal / float(num_equal + num_different)
    return rate, num_equal, num_different # rate.item()

def load_data(dirs):
    print("Reading training images...")
    all_imgs = []
    all_img_names = []
    for i in range(len(dirs)):
        imgs, img_names = get_all_images_dir(dirs[i])
        all_images.append(imgs)
        all_img_names.append(img_names)

    print("Creating data...")
    X = []
    Y = []
    for i in imgs:
        X.append(np.array(img))
    for n in range(len(X)):
        Y.append(np.ones((X[n].shape[0],))*(n+1))

    return X, Y


import torch.nn as nn
import torch.nn.functional as F

cli = argparse.ArgumentParser()
cli.add_argument("--train", nargs="*", type=str)
cli.add_argument("--test", nargs="*", type=str)

args = cli.parse_args()

print("Training directories", args.train)
print("Testing directories", args.test)

train_dirs = args.train
test_dirs = args.test

X_1, Y_1 = load_data(train_dirs)
X_2, Y_2 = load_data(test_dirs)

X_train = torch.Tensor(np.vstack(X_1))
Y_train = Variable(torch.Tensor.long(torch.Tensor(np.vstack(Y_1))))

X_test = torch.Tensor(np.vstack(X_2))
Y_test = Variable(torch.Tensor.long(torch.Tensor(np.vstack(Y_2))))

''' Prepare Training Data  '''
# Read training images
print("Reading training images...")

'''
gr_imgs, gr_img_names = get_all_images_dir("hw6_data/train/grass")
oc_imgs, oc_img_names = get_all_images_dir("hw6_data/train/ocean")
rc_imgs, rc_img_names = get_all_images_dir("hw6_data/train/redcarpet")
rd_imgs, rd_img_names = get_all_images_dir("hw6_data/train/road")
wf_imgs, wf_img_names = get_all_images_dir("hw6_data/train/wheatfield")
'''

# Parse images into feature vectors
print("Creating training data...")

'''
X_gr = np.array(gr_imgs)
Y_gr = np.ones((X_gr.shape[0],))*1

X_oc = np.array(oc_imgs)
Y_oc = np.ones((X_oc.shape[0],))*2

X_rc = np.array(rc_imgs)
Y_rc = np.ones((X_rc.shape[0],))*3

X_rd = np.array(rd_imgs)
Y_rd = np.ones((X_rd.shape[0],))*4

X_wf = np.array(wf_imgs)
Y_wf = np.ones((X_wf.shape[0],))*5
'''


'''
X_train = torch.Tensor(np.concatenate((X_gr, X_oc, X_rc, X_rd, X_wf)))
Y_train = Variable(torch.Tensor.long(torch.Tensor(np.concatenate((Y_gr, Y_oc, Y_rc, Y_rd, Y_wf)))))
'''

''' Prepare Test Data  '''
# Read training images
'''
print("Reading test images...")
gr_imgs, gr_img_names = get_all_images_dir("hw6_data/test/grass")
oc_imgs, oc_img_names = get_all_images_dir("hw6_data/test/ocean")
rc_imgs, rc_img_names = get_all_images_dir("hw6_data/test/redcarpet")
rd_imgs, rd_img_names = get_all_images_dir("hw6_data/test/road")
wf_imgs, wf_img_names = get_all_images_dir("hw6_data/test/wheatfield")

# Parse images into feature vectors
print("Creating test data...")
X_gr = np.array(gr_imgs)
Y_gr = np.ones((X_gr.shape[0],))*1

X_oc = np.array(oc_imgs)
Y_oc = np.ones((X_oc.shape[0],))*2

X_rc = np.array(rc_imgs)
Y_rc = np.ones((X_rc.shape[0],))*3

X_rd = np.array(rd_imgs)
Y_rd = np.ones((X_rd.shape[0],))*4

X_wf = np.array(wf_imgs)
Y_wf = np.ones((X_wf.shape[0],))*5
print(X_gr.shape)

X_test = torch.Tensor(np.concatenate((X_gr, X_oc, X_rc, X_rd, X_wf)))
Y_test = Variable(torch.Tensor.long(torch.Tensor(np.concatenate((Y_gr, Y_oc, Y_rc, Y_rd, Y_wf)))))
'''

n_train, n_valid, n_test = 4000, 1000, 1000
#n_train, n_valid, n_test = 100, 100, 100

'''  Create network  '''
N0 = 259200
N1 = 250
N2 = 25
Nout = 6

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
optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)

#  Print a summary of the network.  Notice that this only shows the layers
print(net)

#  Print a s
params = list(net.parameters())
f = open("nn_weights.txt", "w+")
for p in params:
    f.write(p.size())
print(params[0].size()) # The parameter holding the layer 1 weight matrix
print(params[1].size()) # ... the layer 1 bias vector
print(params[2].size()) # ... the layer 2 weight matrix
print(params[3].size()) # ... the layer 3 bias vector
print(params[4].size()) # ... the layer 4 weight vector
print(params[5].size()) # ... the layer 5 bias vector

'''  Training  '''
#  Set parameters to control the process
epochs = 100
batch_size = 64
n_batches = int(np.ceil(n_train / batch_size))
learning_rate = 1e-6

#  Compute an initial loss using all of the validation data.
#
#  A couple of notes are important here:
#  (1) X_valid contains all of the validation input, with each validation
#      data instance being a row of X_valid
#  (2) Therefore, pred_Y_valid is a Variable containing the output layer
#      activations for each of the validation inputs.
#  (3) This is accomplished through the function call net(X_valid), which in
#      turn calls the forward method under the hood to figure out the flow of
#      the data and activations in the network.
#pred_Y_valid = net(X_valid)
#valid_loss = criterion(pred_Y_valid, Y_valid)
#print("Initial loss: %.5f" % valid_loss.item())

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
        #print(pred_Y)
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
    acc = num_eq / float(num_eq + num_diff)
    print('Test success rate:', acc)
    test_acc.append(acc)
#    pred_Y_train = net(X_train)
#    loss = criterion(pred_Y_train, Y_train)
#    print('Final training loss is %.5f' %loss.item())
#
#    pred_Y_test = net(X_test)
#    test_loss = criterion(pred_Y_test, Y_test)
#    print("Final test loss: %.5f" % test_loss.item())


#    acc = success_rate(pred_Y_test, Y_test)
#    print('Test success rate:', success_rate(pred_Y_test, Y_test))
#    test_acc.append(acc)
#
plt.plot(train_acc, color="green")
plt.plot(test_acc, color="red")
plt.show()

        #  Complete the mini-batch by actually updating the parameters.
#        for param in net.parameters():
#            param.data -= learning_rate * param.grad.data

#    #  Print validation loss every 10 epochs
#    if ep != 0 and ep%10 == 0:
#        pred_Y_valid = net(X_valid)
#        valid_loss = criterion(pred_Y_valid, Y_valid)
#        print("Epoch %d loss: %.5f" %(ep, valid_loss.item()))

#  Compute and print the final training and test loss
#  function values

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
from scipy.signal import find_peaks
import imutils
import matplotlib.pyplot as plt
import math

def get_contour_precedence(contour, cols):
	tolerance_factor = 100
	origin = cv2.boundingRect(contour)
	return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

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

def normalize_gray(im_gr):
	im_norm = np.absolute(im_gr.copy())
	(minVal, maxVal) = (np.min(im_norm), np.max(im_norm))
	im_norm = (255 * ((im_norm - minVal) / (maxVal - minVal)))
	im_norm = im_norm.astype("uint8")
	return im_norm

def segment_words(img_in, rect=(0,0,0,0)):
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 9))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	img = imutils.resize(img_in, height=32)

	wy = img.shape[0]/img_in.shape[0]
	wx = img.shape[1]/img_in.shape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	# compute the Scharr gradient of the tophat image, then scale
	# the rest back into the range [0, 255]
	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
	gradMag = np.sqrt(gradX**2 + gradY**2)

	gradMag = normalize_gray(gradMag)

	# plt.imshow(gradMag, cmap="gray")
	# plt.show()

	thresh = cv2.threshold(gradMag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	# apply a closing operation using the rectangular kernel to help
	# cloes gaps in between credit card number digits, then apply
	# Otsu's thresholding method to binarize the image


	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

	# apply a second closing operation to the binary image, again
	# to help close gaps between credit card number regions
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

	# plt.imshow(thresh, cmap="gray")
	# plt.show()

	# find contours in the thresholded image, then initialize the
	# list of digit locations
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts.sort(key=lambda x:get_contour_precedence(x, thresh.shape[1]))
	locs = []

	# loop over the contours
	for (i, c) in enumerate(cnts):
		# compute the bounding box of the contour, then use the
		# bounding box coordinates to derive the aspect ratio
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h);
		locs.append((int(x//wx + rect[0]), int(y//wy + rect[1]), int(w//wx), int(h//wy)))

	# plt.imshow(thresh, cmap="gray")
	# plt.show()

	return locs


def skeletonize(img_th):
	size = np.size(img_th)
	skel = np.zeros(img_th.shape,np.uint8)

	img_thresh = img_th.copy()*255
	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	done = False

	while( not done):
		eroded = cv2.erode(img_thresh,element)
		temp = cv2.dilate(eroded,element)
		temp = cv2.subtract(img_thresh,temp)
		skel = cv2.bitwise_or(skel,temp)
		img_thresh = eroded.copy()

		zeros = size - cv2.countNonZero(img_thresh)
		if zeros==size:
			done = True

			plt.imshow(skel)
			plt.show()

	return skel

def word_hist(img_line):
	print(img_line.shape)
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	gray = cv2.cvtColor(img_line, cv2.COLOR_BGR2GRAY)

	im_l = imutils.resize(gray, height=100)

	# Binarize the grayscale image using adaptive thresholding
	thresh = cv2.threshold(im_l, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# Skeletonize the Letters
	skel = cv2.ximgproc.thinning(thresh*255)

	# Calculate histogram of each slice along X axis
	hist = np.sum(skel/255, axis=0).reshape(-1,1)

	# Filter the histogram using bilateral filtering to preserve
	# the sharp edges in the histogram
	# hist = cv2.bilateralFilter(hist.astype(np.uint8),15,150,150)
	# hist = cv2.bilateralFilter(hist.astype(np.uint8),15,150,150)
	# hist = cv2.bilateralFilter(hist.astype(np.uint8),15,150,150)
	hist = cv2.GaussianBlur(hist.astype(np.uint8), (7,1), 0)
	hist = cv2.GaussianBlur(hist.astype(np.uint8), (7,1), 0)
	hist = cv2.GaussianBlur(hist.astype(np.uint8), (7,1), 0)
	# min, p = find_peaks(-hist.reshape((-1,)), distance=7, plateau_size=[0,10000])
	# print(min)

	min = np.where(hist.reshape((-1)) < 2, 1, 0).reshape((-1)).astype(np.bool)
	min_arg = np.array(range(min.shape[0]))[min]
	print(min_arg)

	if (min_arg.shape[0] > 0):
		seg_cols = []
		vals = []
		r_min, r_max = min_arg[0], min_arg[0]
		for x in min_arg.tolist():
			if (x > r_max + 8):
				seg_cols.append((r_min, r_max, vals))
				r_min, r_max = x, x
				vals = [x]
			else:
				r_max = x
				vals.append(x)

		seg_cols.append((r_min, r_max, vals))
		print(seg_cols)

		avg_cuts = []
		for seg_col in seg_cols:
			r_min, r_max, vals = seg_col
			avg_cut = int(np.average(np.array(vals)))
			avg_cuts.append(avg_cut)

		print(avg_cuts)

		im_seg = thresh.copy()
		for x in avg_cuts:
			cv2.line(im_seg, (x, 0), (x, im_seg.shape[0]), 1, 1)
		# Display the results of the segmentation
		f, xarr = plt.subplots(3, sharex=True)
		xarr[0].imshow(im_l)
		xarr[1].imshow(skel)
		xarr[2].imshow(im_seg)
		xarr[2].plot(range(hist.shape[0]), hist)
		f.subplots_adjust(hspace=0)
		for ax in xarr:
			ax.label_outer()
		plt.show()

	# thresh = cv2.erode(thresh, sqKernel, iterations=2)
	#
	# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
	# plt.imshow(thresh, cmap="gray")
	# plt.show()

def extract_regions(img_in):
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	img = imutils.resize(img_in, width=300)

	wy = img.shape[0]/img_in.shape[0]
	wx = img.shape[1]/img_in.shape[1]

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	# compute the Scharr gradient of the tophat image, then scale
	# the rest back into the range [0, 255]
	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
	gradMag = np.sqrt(gradX**2 + gradY**2)

	gradMag = normalize_gray(gradMag)

	# plt.imshow(gradMag, cmap="gray")
	# plt.show()

	thresh = cv2.threshold(gradMag, 0, 255,
	cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	# apply a closing operation using the rectangular kernel to help
	# cloes gaps in between credit card number digits, then apply
	# Otsu's thresholding method to binarize the image

	# plt.imshow(thresh, cmap="gray")
	# plt.show()

	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

	# apply a second closing operation to the binary image, again
	# to help close gaps between credit card number regions
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)


	# find contours in the thresholded image, then initialize the
	# list of digit locations
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts.sort(key=lambda x:get_contour_precedence(x, thresh.shape[1]))
	locs = []

	# loop over the contours
	for (i, c) in enumerate(cnts):
		# compute the bounding box of the contour, then use the
		# bounding box coordinates to derive the aspect ratio
		(x, y, w, h) = cv2.boundingRect(c)
		if w < 35 and h < 35:
			continue
		ar = w / float(h);locs.append((int(x//wx), int(y//wy), int(w//wx), int(h//wy)))

	# plt.imshow(thresh, cmap="gray")
	# plt.show()

	return locs

def extract_letters(img, region_rects=None):
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	locs = []
	if (region_rects is not None):
		for i in range(len(region_rects)):
			x,y,w,h = region_rects[i]
			subim = gray[y:y+h, x:x+w]


			thresh = cv2.threshold(subim, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

			# thresh = cv2.erode(thresh, rectKernel, iterations=2)

			thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)

			hist = word_hist(subim)

			# plt.imshow(thresh, cmap="gray")
			# plt.show()
			# find contours in the thresholded image, then initialize the
			# list of digit locations
			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			cnts.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))

			# loop over the contours
			for (i, c) in enumerate(cnts):
				# compute the bounding box of the contour, then use the
				# bounding box coordinates to derive the aspect ratio
				(xl, yl, wl, hl) = cv2.boundingRect(c)
				ar = w / float(h);locs.append((xl+x, yl+y, wl, hl))

	# im_box = img.copy()
	# for (x,y,w,h) in locs:
	#     cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0))


	return locs


if __name__ == "__main__":
	image = cv2.imread(sys.argv[1])

	text_regions = extract_regions(image)

	im_box = image.copy()
	for (x,y,w,h) in text_regions:
		cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0), 2)
#        im_sm = image[y:y+h, x:x+w]
#        reg = extract_letters(im_sm)
	# plt.imshow(im_box[:,:,::-1], cmap="gray")
	# plt.show()

	word_regions = []
	for (x,y,w,h) in text_regions:
		subim = image[y:y+h, x:x+w]
		word_regions += segment_words(subim, rect=(x,y,w,h))

	print(word_regions)

	for (x,y,w,h) in word_regions:
		cv2.rectangle(im_box, (x, y), (x+w, y+h), (0, 255, 0), 2)

	for (x,y,w,h) in word_regions:
		subim = image[y:y+h, x:x+w]
		word_hist(subim)

	plt.imshow(im_box[:,:,::-1], cmap="gray")
	plt.show()
# 	regions = extract_letters(image, regions)
#
# 	im_box = image.copy()
# 	for (x,y,w,h) in regions:
# 		cv2.rectangle(im_box, (x, y), (x+w, y+h), (255, 0, 0))
# #        im_sm = image[y:y+h, x:x+w]
# #        reg = extract_letters(im_sm)
# 	plt.imshow(im_box[:,:,::-1], cmap="gray")
# 	plt.show()



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

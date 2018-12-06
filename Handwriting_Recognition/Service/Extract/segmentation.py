import cv2
import numpy as np
from scipy.ndimage.measurements import label
import random

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

def component():
    return random.randint(0,255)

if __name__ == "__main__":

    image = cv2.imread("test.PNG")
    image = image_resize(image,height = 512)
    image_new = image.copy()

    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

    #dilation
    #kernel = np.ones((100,100), np.uint8)
    kernel = np.ones((20,30), np.uint8)


    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    img_dilation = cv2.erode(img_dilation,None,iterations = 2)

    #find contours
    im2,ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #sort contours
    ctrs.sort(key=lambda x:get_contour_precedence(x, image.shape[1]))

    for i, ctr in enumerate(ctrs):

        # Get bounding box
        rect = cv2.minAreaRect(ctr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        img_line = np.zeros(gray.shape)
        cv2.fillConvexPoly(img_line,box,255)

        # Get component
        roi = np.zeros(image.shape)
        gray_roi = np.zeros(gray.shape)
        roi[img_line==255] = image[img_line==255]
        gray_roi[img_line==255] = thresh[img_line==255]

        # Get min, max, and mean x,y value of each component
        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(gray_roi.transpose(), structure)
        means = []
        minimum = []
        maximum = []
        for j in range(ncomponents):
            means.append(np.mean(np.argwhere(labeled==j+1),axis=0))
            minimum.append(np.min(np.argwhere(labeled==j+1),axis=0))
            maximum.append(np.max(np.argwhere(labeled==j+1),axis=0))

        # combine characters with multiple components
        new_labeled = np.zeros(labeled.shape)
        newncomponents = 1
        if ncomponents == 1:
            new_labeled[labeled==1]=newncomponents
        for j in range(ncomponents-1):
            diff = means[j+1][0] - means[j][0]
            if diff < 10:
                new = j
                new_labeled[labeled==j+1]=newncomponents
            else:

                new_labeled[labeled==j+1]=newncomponents
                newncomponents += 1

        # Label all characters with multiple components
        diff = means[j][0] - means[j-1][0]
        j+=1
        if diff < 10:
            new = j
            new_labeled[labeled==j+1]=newncomponents
        else:
            new_labeled[labeled==j+1]=newncomponents
            newncomponents += 1
        labeled = new_labeled
        ncomponents = newncomponents

        # Output all characters
        for j in range(ncomponents):
            roi[labeled.transpose()==j+1]=[component(),component(),component()]
            test = np.ones(roi.shape)*255
            test[labeled.transpose()==j+1] = 0
            #cv2.imshow(str(i)+"-"+str(j)+".png",test)
            cv2.imwrite(str(i)+"-"+str(j)+".png",test)

        # draw on original image
        image[img_line==255] = roi[img_line==255]
        cv2.drawContours(image,[box],0,(0,0,255),2)

    cv2.imshow('marked areas',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

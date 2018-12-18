import cv2
import numpy as np
from scipy.ndimage.measurements import label
from scipy.spatial import KDTree
import random
import sys

class binTree:
    def __init__(self, val, mean, minimum, maximum):
        self.val = val
        self.line = [val]
        self.meany = [mean[0]]
        self.meanx = [mean[1]]
        self.minimum = [minimum]
        self.maximum = [maximum]
        self.left = None
        self.right = None
        
    def addComponent(self, val, mean, minimum, maximum):
        dist = distance((self.meany[-1],self.meanx[-1]),mean)
        mindist = self.distance(mean, dist, minimum, maximum)
        if mindist == dist:
            if self.minimum[-1] <= maximum and self.maximum[-1] >= minimum:
                self.line.append(val)
                self.meany.append(mean[0])
                self.meanx.append(mean[1])
                self.minimum.append(minimum)
                self.maximum.append(maximum)
            elif self.minimum[-1] >= minimum and self.maximum[-1] <= maximum:
                self.line.append(val)
                self.meany.append(mean[0])
                self.meanx.append(mean[1])
                self.minimum.append(minimum)
                self.maximum.append(maximum)
            elif self.minimum[-1] <= minimum and self.maximum[-1] >= minimum:
                self.line.append(val)
                self.meany.append(mean[0])
                self.meanx.append(mean[1])
                self.minimum.append(minimum)
                self.maximum.append(maximum)
            elif self.minimum[-1] <= maximum and self.maximum[-1] >= maximum:
                self.line.append(val)
                self.meany.append(mean[0])
                self.meanx.append(mean[1])
                self.minimum.append(minimum)
                self.maximum.append(maximum)
            else:
                if self.meany[-1] < mean[0]:
                    if self.left is None:
                        self.left = binTree(val, mean, minimum, maximum)
                    else:
                        self.left.addComponent(val, mean, minimum, maximum)
                else:
                    if self.right is None:
                        self.right = binTree(val, mean, minimum, maximum)
                    else:
                        self.right.addComponent(val, mean, minimum, maximum)    
        else:
            if self.meany[-1] < mean[0]:
                if self.left is None:
                    self.left = binTree(val, mean, minimum, maximum)
                else:
                    self.left.addComponent(val, mean, minimum, maximum)
            else:
                if self.right is None:
                    self.right = binTree(val, mean, minimum, maximum)
                else:
                    self.right.addComponent(val, mean, minimum, maximum)            
                    
    def returnLines(self,arr):
        if not self.right is None:
            self.right.returnLines(arr)
        arr.append(self.line)
        if not self.left is None:
            self.left.returnLines(arr)
            
    def distance(self, mean, dist, minimum, maximum):
        if self.minimum[-1] <= maximum and self.maximum[-1] >= minimum:
            d = distance((self.meany[-1],self.meanx[-1]),mean)
        elif self.minimum[-1] >= minimum and self.maximum[-1] <= maximum:
            d = distance((self.meany[-1],self.meanx[-1]),mean)
        elif self.minimum[-1] <= minimum and self.maximum[-1] >= minimum:
            d = distance((self.meany[-1],self.meanx[-1]),mean)
        elif self.minimum[-1] <= maximum and self.maximum[-1] >= maximum:
            d = distance((self.meany[-1],self.meanx[-1]),mean)
        else:
            d = dist + 1
        leftdist = d + 1
        rightdist = d + 1
        if not self.left is None:
            leftdist = self.left.distance(mean, d, minimum, maximum)
        if not self.right is None:
            rightdist = self.right.distance(mean, d, minimum, maximum)           
        
        return min(dist, leftdist, rightdist)
         
        
def distance(x,y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

# Step 1: Preprocessing of image
# Step 2: Find all connected components
# Step 3: Find mean point of connected components and place them into KDtree
# Step 4: Perform k-nearest neighbors to group letters together
# Step 5: Group letters from step 1 into lines
# Step 6: Large dilation to join i and j letters
# Step 7: Find separation between words???
# Step 8: Segment words
# Step 9: Output final images

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

def color():
    return random.randint(0,255)  

def segment(img_name):
    image = cv2.imread(img_name)
    #image = image_resize(image,height = 512)
    image_new = image.copy()
    
    #grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #binary
    ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    
    structure = np.ones((3, 3), dtype=np.int)
    labeled, ncomponents = label(thresh.transpose(), structure)    
    means = []
    minimum = []
    maximum = []
    for i in range(ncomponents):
        means.append(np.mean(np.argwhere(labeled==i+1),axis=0))
        minimum.append(np.min(np.argwhere(labeled==i+1),axis=0))
        maximum.append(np.max(np.argwhere(labeled==i+1),axis=0))      
    
    
    t = KDTree(means)
    same = dict()
    components = []
    for i in range(ncomponents):
        q = t.query(means[i],3)
        v = q[1][1]
        if components == []:
            components.append([i,v])
        cmpnt_found = False
        for component in components:
            if i in component:
                if not v in component:
                    component.append(v)
                cmpnt_found = True
        if not cmpnt_found:
            for component in components:
                if v in component:
                    if not i in component:
                        component.append(i)
                    cmpnt_found = True    
            if not cmpnt_found:
                components.append([i,v])
            
    #print(components)
    
    new_img = np.zeros(image.shape).astype(np.uint8)
    new_labeled = np.zeros(gray.shape)
    newncomponents = 0
    for component in components:
        #if i != 20:
        #    continue
        newncomponents += 1
        c = [color(),color(),color()]
        for i in component:
            new_img[labeled.transpose() == i+1] = c
            new_labeled[labeled.transpose() == i+1] = newncomponents
            
    means = []
    minimum = []
    maximum = []
    for j in range(newncomponents):
        means.append(np.mean(np.argwhere(new_labeled==j+1),axis=0))
        minimum.append(np.min(np.argwhere(new_labeled==j+1),axis=0))
        maximum.append(np.max(np.argwhere(new_labeled==j+1),axis=0))  
     
    root = binTree(1,means[0],minimum[0][0],maximum[0][0])
    for i in range(1,newncomponents):
        root.addComponent(i+1,means[i],minimum[i][0],maximum[i][0])
     
     
    arr = []
    root.returnLines(arr)
    new_img2 = np.zeros(image.shape).astype(np.uint8)
    """
    newncomponents = 0
    for component in root.line:
        c = [color(),color(),color()]
        new_img2[new_labeled == component] = c
        """
    k = 0
    #lines = []
    for line in arr:
        line_img = np.zeros(gray.shape).astype(np.uint8)
        for component in line:
            line_img[new_labeled == component] = 255
        line_img = line_img.transpose()
        coords = cv2.findNonZero(line_img)
        x, y, w, h = cv2.boundingRect(coords)
        line_img = line_img[y:y+h, x:x+w]     
          
        
        m = max(line_img.shape[0],line_img.shape[1])
        kernel = np.ones((1,m*2), np.uint8)        
            
        img_dilation = cv2.dilate(line_img, kernel, iterations=1) 
        img_widths = np.zeros(img_dilation.shape)
        img_widths[img_dilation == 0] = 255
        
        data = []
        labeled, ncomponents = label(img_widths, structure)  
        if ncomponents ==0:
            continue
        for i in range(ncomponents):
            cpnt = labeled[labeled == i+1]
            data.append(cpnt.shape[0])
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
            
        # Apply KMeans
        if len(data) == 1:
            labels = [[0]]
            centers = data[0]
            split = 0
            char_dist = 0
        else:
            compactness,labels,centers = cv2.kmeans(np.array(data).astype(np.float32),2,None,criteria,10,flags)    
        
            if centers[1] > centers[0]:
                split = 1
                char_dist = centers[0]/labeled.shape[1]
            else:
                split = 0
                char_dist = centers[1]/labeled.shape[1]
        
    
        for i in range(ncomponents):
            if labels[i] != split:
                img_widths[labeled == i+1] =0
                
        word_img = np.zeros(img_widths.shape)
        word_img[img_widths==0] = 255
        
        labeled, ncomponents = label(word_img, structure)
        #words = []
        for i in range(ncomponents):
            word_img[:,:] = 0
            word_img[(labeled==i+1)&(line_img==255)] = 255
            #cv2.imwrite('marked areas'+str(i)+".jpg",word_img)
            # Get min, max, and mean x,y value of each component
            labeled_char, ncomponents_char = label(word_img, structure)    
            means = []
            minimum = []
            maximum = []
            for j in range(ncomponents_char):
                means.append(np.mean(np.argwhere(labeled_char==j+1),axis=0))
                minimum.append(np.min(np.argwhere(labeled_char==j+1),axis=0))
                maximum.append(np.max(np.argwhere(labeled_char==j+1),axis=0))  
        
            # combine characters with multiple components
            n_labeled = np.zeros(labeled_char.shape)
            newncomponents = 1
            if ncomponents_char == 1:
                n_labeled[labeled_char==1]=newncomponents
            for j in range(ncomponents_char-1):
                diff = means[j+1][0] - means[j][0]
                if diff < char_dist:
                    new = j
                    n_labeled[labeled_char==j+1]=newncomponents
                else:
                    
                    n_labeled[labeled_char==j+1]=newncomponents
                    newncomponents += 1   
                
            # Label all characters with multiple components
            diff = means[j][0] - means[j-1][0]
            j+=1
            if diff < char_dist:
                new = j
                n_labeled[labeled_char==j+1]=newncomponents
            else:
                n_labeled[labeled_char==j+1]=newncomponents
                newncomponents += 1         
            labeled_char = n_labeled.transpose()
            ncomponents_char = newncomponents   
            #letters = []
            for j in range(ncomponents_char):
                test = np.zeros(labeled_char.shape).astype(np.uint8)
                test[labeled_char==j+1] = 255
                coords = cv2.findNonZero(test) # Find all non-zero points (text)
                x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
               
                
                rect = test[y:y+h, x:x+w] 
                if (rect.shape[0]==0 or rect.shape[1]==0):
                    continue
                #letters.append([x,y,w,h])
                #cv2.imshow('marked areas',image_resize(rect,height=256))
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()   
                rect[rect==0]=125
                rect[rect==255]=0
                rect[rect==125]=255
                cv2.imwrite(str(k)+"-"+str(i)+"-"+str(j)+".png",image_resize(rect,height=256))     
            #words.append(letters)
        #lines.append(words)
        k += 1
    #print(lines)    


if __name__ == "__main__":
    img_name = sys.argv[1]
    segment(img_name)
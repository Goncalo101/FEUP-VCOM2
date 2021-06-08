import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt

from tqdm import tqdm

class Vocabulary:
    def __init__(self, nWords):
        self.vocabulary = None
        self.nWords = nWords
        
    def train(self, listOfImages):
        # detector = cv.xfeatures2d.SIFT_create()
        detector = cv.BRISK_create(thresh=30)
        allDescriptors = []
        for name in tqdm(listOfImages):
            img = open_image(name)
            if img is None:
                continue
            keypoints, descriptors = detector.detectAndCompute(img, None)

            if descriptors is None:
                _detector = cv.BRISK_create(thresh=20)

                keypoints, descriptors = _detector.detectAndCompute(img, None)
                allDescriptors.append((name, descriptors))
                continue
            
            if allDescriptors is None:
                print('Big oof!', name)
                continue

            allDescriptors.append((name, descriptors))

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness,labels,centers = cv.kmeans(np.float32(allDescriptors), self.nWords, None, criteria, 100, cv.KMEANS_PP_CENTERS)
        self.vocabulary = centers

    def whichWord(self, descriptor):
        if self.vocabulary.shape[0] <= 1:
            return -1

        minIndex = 0
        #minDistance = cv.norm(self.vocabulary[0,:]-descriptor,cv.NORM_L2)
        minDistance = np.linalg.norm(self.vocabulary[0,:]-descriptor)

        for i in range(1, self.vocabulary.shape[0]):
            #distance = cv.norm(self.vocabulary[i,:]-descriptor,cv.NORM_L2)
            distance = np.linalg.norm(self.vocabulary[i,:]-descriptor)
            if distance < minDistance:
                minDistance = distance
                minIndex = i

        return minIndex

def open_image(filename):
    image = cv.imread(filename)
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    except:
        print(" --(!) Error reading image ", filename)
        return None
    return image

def draw_keypoints(windowName, image, keypoints, words):
    if len(keypoints) != len(words):
        return

    newImage = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    maxw=0
    for word in words:
        if word>maxw:
            maxw = word

    steps = int(255/(math.log(maxw+1)/math.log(3)))
    colors = []
    for r in range(1,256,steps):
        for g in range(1,256,steps):
            for b in range(1,256,steps):
                colors.append((b,g,r))

    positions = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    for i in range(len(keypoints)):
        cv.circle(newImage, positions[i], 4, (colors[words[i]]), 2)

    # cv.namedWindow( windowName, cv.WINDOW_AUTOSIZE )
    # cv.imshow( windowName, newImage )
    
    plt.imshow(newImage)
    plt.title(windowName)
    plt.xticks([]), plt.yticks([])
    plt.show()
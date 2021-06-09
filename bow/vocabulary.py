from types import new_class
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

class Vocabulary:
    def __init__(self, nWords):
        self.vocabulary = None
        self.nWords = nWords
        self.descriptor_list = []
        
    def train(self, listOfImages):
        # Create feature extractors and keypoint detectors using BRISK 
        # which offers better performance for this dataset than KAZE and SIFT.
        # Using a threshold of 20 (lower than the default 30) significantly lowers 
        # the number of images from where the detector cannot retrieve descriptors.
        detector = cv.BRISK_create(thresh=20)

        self.descriptor_list = []
        for name in tqdm(listOfImages):
            img = open_image(name)
            if img is None:
                continue

            keypoints, img_descriptors = detector.detectAndCompute(img, None)
            
            if img_descriptors is None:
                print(f'No descriptors found for {name}. Skipping')
                continue

            self.descriptor_list.append((name, img_descriptors))

        # Stack descriptors vertically in a numpy array
        print('Stacking descriptors')
        sys.stdout.flush()
        descriptors = self.descriptor_list[0][1]
        for img_path, descriptor in tqdm(self.descriptor_list[1:]):
            descriptors = np.vstack((descriptors, descriptor))

        # Perform k-means clustering on the descriptors, with as many clusters as the number of words
        print('Computing clusters...')
        # --- Scipy
        # descriptors_f = descriptors.astype(float)
        # voc, variance = kmeans(descriptors_f, self.nWords)
        # --- OpenCV
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # compactness,labels,centers = cv.kmeans(np.float32(descriptors), self.nWords, None, criteria, 50, cv.KMEANS_PP_CENTERS)
        # --- scikit-learn
        kmeans = MiniBatchKMeans(n_clusters=self.nWords, random_state=0, batch_size=25)
        kmeans.fit(descriptors)
        print('Done!')

        # Store the centroids for each cluster
        # self.vocabulary = voc
        # self.vocabulary = centers
        self.vocabulary = kmeans.cluster_centers_

    def which_word(self, descriptor):
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
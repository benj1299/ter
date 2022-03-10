from Helper import Helper
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import *
from sklearn.cluster import KMeans
from sklearn.metrics import *
from pathlib import Path

class Ia:
    def __init__(self, path):
        self.path = path
        self.descriptors = []
        self.rois = []

    def extract_roi(self):
        images = Helper.load_images_from_folder(self.path)

        for image in images:
            # Denoising the image
            image = cv2.GaussianBlur(image, (5, 5), 3)
            
            # Binarisation et utilisation de OTSU pour déterminer le seuil automatiquement
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Ne detecte que les flèches noires, il faut modifier le param 2 et 3 pour inverser cela et ajouter l'inverse de l'image
                
            # Contours Detection
            edged = cv2.Canny(binary,10,200)

            contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Get only extreme points of the contours
            
            # Sort the contours by area and define le threshold area min
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            threshold_area_min = cv2.contourArea(contours[0])*0.05

            num = 0
            #descriptors = []
            #rois = []

            for i, c in enumerate(contours):
                x,y,w,h = cv2.boundingRect(c)

                if Helper.is_triangle(c) and cv2.contourArea(c) > threshold_area_min:
                    
                    ROI = binary[y:y+h, x:x+w]

                    # Add white border
                    ROI = np.pad(ROI, pad_width=4, mode='constant', constant_values=255)

                    edged = cv2.Canny(binary,10,200)
                    #edged = cv2.convertScaleAbs(cv2.Laplacian(ROI,cv2.CV_64F))

                    contours2, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    biggest = max(contours2, key=cv2.contourArea)

                    im = np.copy(ROI)

                    #mask = np.ones(ROI.shape[:2], np.uint8)
                    cv2.drawContours(im,[biggest],-1,(0, 0, 255),3)
                
                    # make here the shape detection and filter ROIs

                    self.descriptors.append(Helper.GFD(ROI, 4, 9))
                    self.rois.append(ROI)

                    #cv2.drawContours(ROI, [approx], -1, (0, 255, 0), 3)            
                    cv2.imwrite('./resultats/ROI_{}.png'.format(num), ROI)
                    num += 1

    def make_clustering(self):
        descriptors = np.array(self.descriptors, dtype=object)
        rois = np.array(self.rois, dtype=object)

        x, y, z = descriptors.shape
        X = descriptors.reshape((x,y*z))

        #elbow_method(X)

        model = KMeans(n_clusters=3, init='k-means++')
        labels = model.fit_predict(X)
        plt.scatter(X[:,0], X[:, 1], c=labels)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1])

        #print(model.labels_)
        
        for i in range (0, 2):
            for num, image in enumerate(rois[model.labels_ == i]):
                Path(f'./resultats/class_{i+1}').mkdir(exist_ok=True)
                cv2.imwrite(f'./resultats/class_{i+1}/{num}.png', image)
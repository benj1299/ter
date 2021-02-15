import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import *

class Helper:

    """ 
        Make a numpy array of all images in the folder's path

        Parameters
        ----------
        folder (string) : the folder's path
    
        Returns
        -------
        Numpy array of all images readed by OpenCV

        """
    @staticmethod
    def load_images_from_folder(folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename), 0) # Gray Color converting
            if img is not None:
                images.append(img)
        return np.array(images)

    """ 
        Bresenham's Algorithm
        Produces a list of tuples from start and end

         Parameters
        ----------
            start (tuple of float) : the departure point
            end (tuple of float) : the end point

        Returns
        -------
            Numpy array of all points of the segment

        Tests
        -----
            points1 = get_line((0, 0), (3, 4))
            points2 = get_line((3, 4), (0, 0))
            assert(set(points1) == set(points2))
            print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
            print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    @staticmethod
    def bresenham(start, end):
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
    
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        return np.array(points)

    """
        Generique Fourrier Descriptor

        Argument
        --------
        The opencv image

        Return
        ------
        GFD Signature
    """
    @staticmethod
    def GFD(image, m, n):
        # preprocess the image
        if len(image.shape) > 2:
            image = image.max(axis = 2) / 255
        width = image.shape[1]
        N = width
        maxR = sqrt((((N)//2)**2) + (((N)//2)**2))

        x = np.linspace(-(N-1)//2, (N-1)//2, N )
        y = x
        X, Y = np.meshgrid(x, y)

        radius = np.sqrt(np.power(X, 2) + np.power(Y, 2)) / maxR

        theta = np.arctan2(Y, X)
        theta[theta < 0] = theta[theta < 0] + 2+ np.pi

        FR = np.zeros((m,n))
        FI = np.zeros((m,n))
        FD = np.zeros((m*n,1))

        i = 0
        for rad in range(m):
            for ang in range(n):
                # e^(i * theta) = cos(theta) + i * sin(theta)
                # PF = FR + i * FI

                tempR = image * np.cos(2 * np.pi * rad * radius + ang * theta)
                tempI = image * np.sin(2 * np.pi * rad * radius + ang * theta)
                FR[rad, ang] = np.sum(tempR)
                FI[rad, ang] = np.sum(tempI)
                
                if rad == 0 and ang == 0:
                    FD[i] = sqrt((2* (FR[0,0] * FR[0,0]))) / (np.pi* maxR * maxR)
                else:
                    FD[i] = sqrt((FR[rad, ang] * FR[rad, ang]) + (FI[rad, ang] * FI[rad, ang])) / (math.sqrt((2* (FR[0,0] * FR[0,0]))))
                i = i + 1

        return FD
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from math import *
from sklearn.cluster import KMeans
from sklearn.metrics import *
import shutil

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
        return np.array(images, dtype=object)

    """ 
        Supprime le contenu d'un dossier

        Parameters
        ----------
        folder (string) : the folder's path
    
        Returns
        -------

    """
    @staticmethod
    def auto_remove_results(folder="./resultats"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            return
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Problème de suppression %s. Raisonz: %s' % (file_path, e))

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
        N = image.shape[1]
        maxR = sqrt((((N)//2)**2) + (((N)//2)**2))

        x = np.linspace(-(N-1)//2, (N-1)//2, N)
        y = x
        X, Y = np.meshgrid(x, y)

        radius = np.sqrt(np.power(X, 2) + np.power(Y, 2)) / maxR

        theta = np.arctan2(Y, X)
        theta[theta < 0] = theta[theta < 0] + (2 * np.pi)

        FR = np.zeros((m,n))
        FI = np.zeros((m,n))
        FD = np.zeros((m*n,1))

        i = 0
        for rad in range(m):
            for ang in range(n):
                tempR = image.dot(np.cos(2 * np.pi * rad * radius + ang * theta))
                tempI = image.dot(np.sin(2 * np.pi * rad * radius + ang * theta))
                FR[rad, ang] = np.sum(tempR)
                FI[rad, ang] = np.sum(tempI)
                
                if rad == 0 and ang == 0:
                    FD[i] = sqrt((2* (FR[0,0] * FR[0,0]))) / (np.pi* maxR * maxR)
                else:
                    FD[i] = sqrt((FR[rad, ang] * FR[rad, ang]) + (FI[rad, ang] * FI[rad, ang])) / (sqrt((2* (FR[0,0] * FR[0,0]))))
                i = i + 1
        return FD

    """
        Methode Elbow pour obtenir le nombre de cluster le plus optimal de Kmeans

        Argument
        --------
        Model X à tester

        Return
        ------
        Void
    """
    @staticmethod
    def elbow_method(X):
        inertia = []
        K_range = range(1, 10)
        for k in K_range:
            model = KMeans(n_clusters=k).fit(X)
            inertia.append(model.inertia_)
        plt.plot(K_range, inertia)
        plt.xlabel('nombre de clusters')
        plt.ylabel('cout du model (inertia)')

    """
        Calcule la distance euclidienne de deux vecteurs

        Argument
        --------
        Vecteurs

        Return
        ------
        Distance en Float
    """
    @staticmethod
    def euclidean_distance(vec1, vec2):
        return np.sum(np.sqrt((vec1 - vec2)**2))

    @staticmethod
    def distance(a,b):
        x1 = a[0]
        x2 = b[0]
        y1 = a[1]
        y2 = b[1]
        return sqrt((x1-x2)**2+(y1-y2)**2)

    @staticmethod
    def sumDistance(a,points):
        sum = 0
        for point in points:
            sum += distance(a,point)
        return sum

    @staticmethod
    def heron(a,b,c):
        s = (a + b +c)/2 
        try:
            res = sqrt(s*(s-a)*(s-b)*(s-c))
        except ValueError:
            return 0
        else:
            return res

    @staticmethod
    def equaDist(a,b,c):
        return (isclose(distance(a,b),distance(b,c),rel_tol = 0.07))

    @staticmethod
    def isIsoceles(points,nbrPoints = 3):
        if(nbrPoints == 3):
            a = points[0][0]
            b = points[1][0]
            c = points[2][0]
            #on suppose qu'on aura 3 points en entrée
            #isocele en b
            if (equaDist(a,b,c) and not (equaDist(b,c,a))):
                return True
            #isocele en c
            elif (equaDist(b,c,a) and not (equaDist(c,a,b))):
                return True
            #isocele en a
            elif (equaDist(c,a,b) and not (equaDist(a,b,c))):
                return True
            else :
                return False
        elif(nbrPoints == 4):
            l1 = points[1:4,0]   
            sum_d1 = sumDistance(points[0][0],l1)

            l2 = points[0:1,0]+points[2:4,0]
            sum_d2 = sumDistance(points[1][0],l2)

            l3 = points[0:2,0]+points[3:4,0]
            sum_d3 = sumDistance(points[2][0],l3)

            l4 = points[0:3,0]
            sum_d4 = sumDistance(points[3][0],l4)
            
            dl = [sum_d1,sum_d2,sum_d3,sum_d4]

            if(dl.index(min(dl)) == 0):
                a = points[3][0]
                b = points[1][0]
                c = points[2][0]
                #isocele en b
                if (equaDist(a,b,c) and not (equaDist(b,c,a))):
                    return True
                #isocele en c
                elif (equaDist(b,c,a) and not (equaDist(c,a,b))):
                    return True
                #isocele en a
                elif (equaDist(c,a,b) and not (equaDist(a,b,c))):
                    return True
                else :
                    return False

            elif(dl.index(min(dl)) == 1):
                a = points[0][0]
                b = points[2][0]
                c = points[3][0]
                #isocele en b
                if (equaDist(a,b,c) and not (equaDist(b,c,a))):
                    return True
                #isocele en c
                elif (equaDist(b,c,a) and not (equaDist(c,a,b))):
                    return True
                #isocele en a
                elif (equaDist(c,a,b) and not (equaDist(a,b,c))):
                    return True
                else :
                    return False
                
            elif(dl.index(min(dl)) == 2):
                a = points[0][0]
                b = points[1][0]
                c = points[3][0]
                #isocele en b
                if (equaDist(a,b,c) and not (equaDist(b,c,a))):
                    return True
                #isocele en c
                elif (equaDist(b,c,a) and not (equaDist(c,a,b))):
                    return True
                #isocele en a
                elif (equaDist(c,a,b) and not (equaDist(a,b,c))):
                    return True
                else :
                    return False

            elif(dl.index(min(dl)) == 3):
                a = points[0][0]
                b = points[1][0]
                c = points[2][0]
                #isocele en b
                if (equaDist(a,b,c) and not (equaDist(b,c,a))):
                    return True
                #isocele en c
                elif (equaDist(b,c,a) and not (equaDist(c,a,b))):
                    return True
                #isocele en a
                elif (equaDist(c,a,b) and not (equaDist(a,b,c))):
                    return True
                else :
                    return False
            else :
                print("Error")
                return False
        else :
            print("Error")
            return False    

    @staticmethod
    def heronPoints(points,nbrPoints = 3):
        #on suppose qu'on aura 3 points en entrée
        if(nbrPoints == 3):
            d1 = distance(points[0][0],points[1][0])
            d2 = distance(points[1][0],points[2][0])
            d3 = distance(points[2][0],points[0][0])

            return heron(d1,d2,d3)   
        
        #on suppose qu'on aura 4 points en entrée
        elif(nbrPoints == 4):
            l1 = points[1:4,0]   
            sum_d1 = sumDistance(points[0][0],l1)

            l2 = points[0:1,0]+points[2:4,0]
            sum_d2 = sumDistance(points[1][0],l2)

            l3 = points[0:2,0]+points[3:4,0]
            sum_d3 = sumDistance(points[2][0],l3)

            l4 = points[0:3,0]
            sum_d4 = sumDistance(points[3][0],l4)

            dl = [sum_d1,sum_d2,sum_d3,sum_d4]

            if(dl.index(min(dl)) == 0):
                d1 = distance(points[1][0],points[2][0])
                d2 = distance(points[2][0],points[3][0])
                d3 = distance(points[3][0],points[1][0])

                return heron(d1,d2,d3)

            elif(dl.index(min(dl)) == 1):
                d1 = distance(points[0][0],points[2][0])
                d2 = distance(points[2][0],points[3][0])
                d3 = distance(points[3][0],points[0][0])

                return heron(d1,d2,d3)

            elif(dl.index(min(dl)) == 2):
                d1 = distance(points[0][0],points[1][0])
                d2 = distance(points[1][0],points[3][0])
                d3 = distance(points[3][0],points[0][0])

                return heron(d1,d2,d3)

            elif(dl.index(min(dl)) == 3):
                d1 = distance(points[0][0],points[1][0])
                d2 = distance(points[1][0],points[2][0])
                d3 = distance(points[2][0],points[0][0])

                return heron(d1,d2,d3)
            else :
                print("Error")
                return -1
        else :
            print("Error")
            return -1

    """
        Calcule un contour pour définir si celui-ci est un triangle ou non

        Argument
        --------
        Contour opencv d'une image

        Return
        ------
        Vrai ou Faux
    """
    @staticmethod
    def is_triangle(contour):
            epsilon = 0.07 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            return len(approx) >= 3 and len(approx) <= 5
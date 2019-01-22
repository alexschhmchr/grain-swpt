import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def draw_Keypoints():
    imgOne = cv.imread('DSC_1043.JPG', 0)
    imgTwo = cv.imread('DSC_1044.JPG', 0)
    imgThree = cv.imread('DSC_0923.JPG', 0)
    imgFour = cv.imread('DSC_0924.JPG', 0)

    array_Of_Grain = [imgOne, imgTwo, imgThree, imgFour]

    # Initiate STAR detector
    orb = cv.ORB_create()
    i = 0
    while i < len(array_Of_Grain):
        # find the keypoints with ORB
        kp = orb.detect(array_Of_Grain[i], None)

        # compute the descriptors with ORB
        kp, des = orb.compute(array_Of_Grain[i], kp)

        # draw only keypoints location,not size and orientation
        img2 = cv.drawKeypoints(array_Of_Grain[i], kp, outImage=np.array([]), color=(255, 0, 0), flags=0)

        plt.imshow(img2), plt.show()

        i = i + 1



draw_keypoints()


import numpy as np
import cv2

def combineImages(array):
    finalImage=array[0];
    array.pop(0)
    for element in array:
        finalImage = np.concatenate((finalImage, element), axis=0)
    return finalImage

def saveImage(npImage, imageName, extension):
    cv2.imwrite(imageName +  extension, npImage)
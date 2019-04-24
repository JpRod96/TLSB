import numpy as np
import cv2
import os

def combineImages(array):
    finalImage=array[0];
    array.pop(0)
    for element in array:
        finalImage = np.concatenate((finalImage, element), axis=0)
    return finalImage

def saveImage(npImage, imageName, extension):
    cv2.imwrite(imageName +  extension, npImage)

def saveImageToPath(npImage, imageName, extension, path):
    cv2.imwrite(os.path.join(path , imageName +  extension), npImage)

def getPathOfVideoDirectory(path):
    tokens = path.split("/")
    tokensSize = len(tokens)
    tokens.pop(tokensSize-1)
    directory = ""
    for token in tokens:
        directory = directory + token + "/"
    return directory

def getLastTokenOfPath(pathString):
    tokens = pathString.split("/")
    tokensSize = len(tokens)
    file = tokens[tokensSize-1]
    tokens = file.split(".")
    return tokens
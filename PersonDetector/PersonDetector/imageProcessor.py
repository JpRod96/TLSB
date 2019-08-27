import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
import util

class ImageProcessor:
    pictureSize = None
    JPG_EXTENSION = "jpg"

    def __init__(self, pictureSize):
        self.pictureSize = pictureSize
        self.edgeDetector = EdgeDetector(pictureSize)
    
    def isGivenFileJPGFile(self, file):
        finalToken = util.getLastTokenOfPath(file)
        return finalToken[1] == self.JPG_EXTENSION
    
    def isGivenPathADir(self, path):
        finalToken = util.getLastTokenOfPath(path)
        return len(finalToken) == 1
    
    def grayScaleImagesOf(self, path):
        if self.isGivenPathADir(path) :
            imageFiles = self.getImageFilesFromDirectory(path)
            for imageFile in imageFiles:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.toGrayscale(img)
                cv2.imwrite(self.combineName(path + "/" + imageFile, "grey"), img)
        else :
            img = cv2.imread(path, -1)
            img = self.edgeDetector.toGrayscale(img)
            cv2.imwrite(self.combineName(path, "grey"), img)
    
    def blurredEdgeImagesOf(self, path, kH, kw):
        if self.isGivenPathADir(path) :
            imageFiles = self.getImageFilesFromDirectory(path)
            for imageFile in imageFiles:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight= kH, kernelWidth=kw)
                cv2.imwrite(self.combineName(path + "/" + imageFile, "edges"), img)
        else :
            img = cv2.imread(path, -1)
            img = self.edgeDetector.getImageBluryEdgesFromNumpy(img, kernelHeight= kH, kernelWidth=kw)
            cv2.imwrite(self.combineName(path, "edges"), img)
    
    def edgeImagesOf(self, path, kH, kw):
        if self.isGivenPathADir(path) :
            imageFiles = self.getImageFilesFromDirectory(path)
            for imageFile in imageFiles:
                img = cv2.imread(path + "/" + imageFile, -1)
                img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight= kH, kernelWidth=kw)
                cv2.imwrite(self.combineName(path + "/" + imageFile, "edges"), img)
        else :
            img = cv2.imread(path, -1)
            img = self.edgeDetector.getImageEdgesFromNumpy(img, kernelHeight= kH, kernelWidth=kw)
            cv2.imwrite(self.combineName(path, "edges"), img)
        
    
    def combineName(self, path, toCombine):
        fileName, extension = util.getLastTokenOfPath(path)[:2]
        newFileName = fileName + toCombine + "." + extension
        path = util.getPathOfVideoDirectory(path)
        return path+newFileName

    def getImageFilesFromDirectory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        imageFiles = list(filter(lambda x: self.isGivenFileJPGFile(x) , files))
        return imageFiles
    
    
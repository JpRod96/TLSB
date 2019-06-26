import numpy as np
import cv2
import util
from scipy import ndimage
from random import randrange, choice

class VideoCutter:
    PHOTO="photo"
    JPG_EXTENSION=".jpg"

    def cutVideo(self, videoPath, outFramesNumber, rotate, typeOfCut):
        cap = cv2.VideoCapture(videoPath)
        videoName = util.getLastTokenOfPath(videoPath)
        print("Video " + videoName[0] + " loaded")
        #numero total de frames
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        constantForProbablity = 5
        cutOn = 30
        cont = 0
        listOfFrames = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            cont += 1

            if cont >= cutOn:
                #Rotar imagen
                if rotate:
                    rotatedFrame = ndimage.rotate(frame, 270)
                    listOfFrames.append(rotatedFrame)
                else:
                    listOfFrames.append(frame)
                # Corte en el frame:
                if typeOfCut == "Constant":
                    cutOn = cont + 10
                if typeOfCut == "Probabilistic":
                    cutOn = frameCount/outFramesNumber + cont
                    cutOn = self.getRandomCut(cutOn, constantForProbablity)
                #if len(listOfFrames) == outFramesNumber:
                #    break
            else:
                if cutOn >= frameCount:
                    break
            
        cap.release()
        return listOfFrames

    def saveFrames(self, listOfFrames):
        numberOfFrame = 1
        for photo in listOfFrames:
            imageName = self.PHOTO + str(numberOfFrame) + self.JPG_EXTENSION;
            cv2.imwrite (imageName, photo)
            numberOfFrame += 1

    def getRandomNumber(self, constantForProbablity):
        randomNumber = randrange(constantForProbablity)
        return randomNumber

    def getRandomOperation(self):
        operation = choice(["Suma", "Resta"])
        return operation

    def getRandomCut(self, CutOn, constantForProbablity):
        randomOperation = self.getRandomOperation()
        randomNumber = self.getRandomNumber(constantForProbablity)
        if randomOperation == "Suma":
            randomCut = CutOn + randomNumber
        if randomOperation == "Resta":
            randomCut = CutOn - randomNumber
        return randomCut
    
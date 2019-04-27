import numpy as np
import cv2
import util
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
        cutOn = 0
        cont = 0
        listOfFrames = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            cont += 1

            if cont >= cutOn:
                #Rotar imagen
                if rotate:
                    rotatedFrame = self.rotateFrame(frame)
                    listOfFrames.append(rotatedFrame)
                else:
                    listOfFrames.append(frame)
                # Corte en el frame:
                if typeOfCut == "Constant":
                    cutOn = frameCount/outFramesNumber + cont
                    print("CUT ON: " + str(cutOn))
                if typeOfCut == "Probabilistic":
                    cutOn = frameCount/outFramesNumber + cont
                    cutOn = self.getRandomCut(cutOn, constantForProbablity)
                if len(listOfFrames) == outFramesNumber:
                    break
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

    def rotateFrame(self, frame):
        rows,cols,dimensions = frame.shape
        rotationParameters = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        rotatedFrame = cv2.warpAffine(frame,rotationParameters,(cols,rows))
        return rotatedFrame

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
    
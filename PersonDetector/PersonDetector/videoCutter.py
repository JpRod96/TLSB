import numpy as np
import cv2
import util

class VideoCutter:
    PHOTO="photo"
    JPG_EXTENSION=".jpg"

    def cutVideo(self, videoPath, outFramesNumber):
        cap = cv2.VideoCapture(videoPath)
        videoName = util.getLastTokenOfPath(videoPath)
        print("Video " + videoName[0] + " loaded")
        #numero total de frames
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cutOn = 0
        cont = 0
        listOfFrames = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            cont += 1
            if cont >= cutOn:
                listOfFrames.append(frame)
                # Corte en el frame:
                cutOn = frameCount/outFramesNumber + cont
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
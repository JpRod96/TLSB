from videoProcessorI import VideoProcessorI
#import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import argrelmax
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
from PersonDetector import PersonDetector

class VideoMotionProcessor(VideoProcessorI):

    criticalFramesPath = "criticalFrames"
    iterations = 4

    def __init__(self, finalPicSize):
        self.edgeDetector = EdgeDetector(finalPicSize)
        self.detector = PersonDetector()

    def process(self, videoPath):
        record, frames = self.cutVideo(videoPath)
        
        maximaValues = record
        maximaIndexes = None
        globalIndexes = np.arange(0, len(maximaValues), 1)

        #plt.plot(maximaValues)
        #plt.ylabel('Alike percentage')
        #plt.show()

        for x in range(1, self.iterations):
            maximaIndexes = argrelmax(maximaValues)[0]
            maximaValues = maximaValues[maximaIndexes]
            globalIndexes = globalIndexes[maximaIndexes]

        print(maximaIndexes)
        print(maximaValues)
        print(globalIndexes)
        self.saveCriticalFrames(frames, globalIndexes)

    def saveCriticalFrames(self, frames, indexes):
        for index in indexes:
            fileName = self.criticalFramesPath +"/"+ str(index) +  ".jpg"
            cv2.imwrite(fileName, frames[index])

    def cutVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        print("Video " + videoPath + " loaded")
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(frameCount)
        lastFrame = None
        onGoingFrame = None
        listOfFrames = []
        cont = 0
        forPlotting = []

        while(cap.isOpened()):
            ret, frame = cap.read()
            lastFrame = onGoingFrame
            onGoingFrame = frame
            listOfFrames.append(frame)
            cont+=1
            if(cont>1):
                match = self.compare(lastFrame, onGoingFrame)
                forPlotting.append(match)
                self.printProgressBar(cont, frameCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if(cont >= frameCount):
                break
        cap.release()
        return np.array(forPlotting), listOfFrames

    def compare(self, pic1, pic2):
        sift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = sift.detectAndCompute(pic1, None)
        kp_2, desc_2 = sift.detectAndCompute(pic2, None)

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6*n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        matchPercentage = len(good_points) / number_keypoints * 100
        return matchPercentage

#make utilitary

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
            if iteration == total: 
                print()
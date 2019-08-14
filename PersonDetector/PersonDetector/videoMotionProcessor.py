from videoProcessorI import VideoProcessorI
#import matplotlib.pyplot as plt
import util
import cv2
import numpy as np
from scipy.signal import argrelmax
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
from PersonDetector import PersonDetector

class VideoMotionProcessor(VideoProcessorI):
    iterations = 4
    MAX_ALIKE_PERCENTAGE = 31

    def __init__(self, finalPicSize, toCombine, framesNr):
        self.edgeDetector = EdgeDetector(finalPicSize)
        self.detector = PersonDetector()
        self.combineImages = toCombine
        self.framesNumber = framesNr

    def process(self, videoPath):
        frames = self.cutVideo(videoPath)
        record = self.getAlikeWeights(frames)
        self.directory = util.getPathOfVideoDirectory(videoPath)
        self.videoName = util.getLastTokenOfPath(videoPath)[0]
        
        maximaIndexes, maximaValues, globalIndexes = self.processForGettingMaxAlike(record)

        print(maximaIndexes)
        print(maximaValues)
        print(globalIndexes)

        globalIndexes = self.postProcessFilter(frames, globalIndexes)

        if(self.combineImages):
            self.combineFrames(frames, globalIndexes)
        else:
            self.saveCriticalFrames(frames, globalIndexes)

    def postProcessFilter(self, criticalFrames, indexes):
        print("Starting post-process filter")

        finalFramesIndexes = []
        compareFrameIndex = indexes[0]
        compareFrame = criticalFrames[compareFrameIndex]
        cont = 0

        for index in range(0, len(indexes) - 1 ):
            globalIndex = indexes[index + 1]
            actualFrame = criticalFrames[globalIndex]

            match = self.compare(compareFrame, actualFrame)
            #print(str(match))

            if(match < self.MAX_ALIKE_PERCENTAGE):
                finalFramesIndexes.append(compareFrameIndex)
            compareFrame = actualFrame
            compareFrameIndex = globalIndex
            cont += 1
            self.printProgressBar(cont, len(indexes) - 1 , prefix = 'Post-Process:', suffix = 'Complete', length = 25)

        finalFramesIndexes.append(compareFrameIndex)
        return finalFramesIndexes

    def processForGettingMaxAlike(self, weigths):
        maximaValues = weigths
        maximaIndexes = None
        globalIndexes = np.arange(0, len(maximaValues), 1)

        #plt.plot(maximaValues)
        #plt.ylabel('Alike percentage')
        #plt.show()

        for x in range(1, self.iterations):
            maximaIndexes = argrelmax(maximaValues)[0]
            maximaValues = maximaValues[maximaIndexes]
            globalIndexes = globalIndexes[maximaIndexes]
        
        return maximaIndexes, maximaValues, globalIndexes

    def processForGettingMinAlike(self, weigths):
        minimaValues = weigths
        minimaIndexes = None
        globalIndexes = np.arange(0, len(minimaValues), 1)

        #plt.plot(minimaValues)
        #plt.ylabel('Alike percentage')
        #plt.show()

        for x in range(1, self.iterations):
            minimaIndexes = argrelmin(minimaValues)[0]
            minimaValues = minimaValues[minimaIndexes]
            globalIndexes = globalIndexes[minimaIndexes]
        
        return minimaIndexes, minimaValues, globalIndexes

    def saveCriticalFrames(self, frames, indexes):
        counter = 1
        for index in indexes:
            fileName = self.directory +"/"+ self.videoName + str(counter) +  ".jpg"
            counter += 1
            cv2.imwrite(fileName, frames[index])
    
    def combineFrames(self, frames, indexes):
        x=1
        edgeImages=[]
        for index in indexes:
            frame = frames[index]
            print("Processing frame number "+ str(x) +"...")
            try:
                treatedImage = self.detector.detectPersonFromNumpy(frame)
                edgeImages.append(self.edgeDetector.getImageEdgesFromNumpy(treatedImage))
                print("Done.\n")
            except:
                print("Human not found on frame number "+ str(x))
            x=x+1
        edgeImages = self.satisfyFramesNumber(edgeImages)
        if(len(edgeImages)>0):
            print("Concatenating all images")
            data = util.combineImages(edgeImages)
            util.saveImageToPath(data, self.videoName + "Edges", ".jpg", self.directory)
            print("Done.")
        else:
            print("Unsuccesful process, theres no human on the video clip "+ self.videoName +"\n")
    
    def satisfyFramesNumber(self, framesArray):
        if(self.framesNumber > 0):
            if(self.framesNumber > len(framesArray)):
                self.increaseArrayLength(framesArray)
            elif(self.framesNumber < len(framesArray)):
                self.decreaseArrayLength(framesArray)
        return framesArray

    def increaseArrayLength(self, array):
        difference = self.framesNumber - len(array)
        lastIndex = len(array)-1
        lastFrame = array[lastIndex]
        for x in range(0, difference):
            array.append(lastFrame)

    def decreaseArrayLength(self, array):
        difference = len(array) - self.framesNumber
        for x in range(0, difference):
            array.pop()

    def cutVideo(self, videoPath):
        cap = cv2.VideoCapture(videoPath)
        print("Video " + videoPath + " loaded")
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(str(frameCount) + " frames to process...")
        listOfFrames = []
        cont = 0
        while(cap.isOpened() and cont < frameCount):
            ret, frame = cap.read()
            listOfFrames.append(frame)
            cont += 1
        cap.release()
        return listOfFrames
    
    def getAlikeWeights(self, frames):
        frameCount = len(frames)
        weights = []
        cont = 0
        lastFrame = None
        onGoingFrame = None

        for index in range(0, frameCount):
            frame = frames[index]
            lastFrame = onGoingFrame
            onGoingFrame = frame
            cont+=1
            if(cont > 1):
                match = self.compare(lastFrame, onGoingFrame)
                weights.append(match)
                self.printProgressBar(cont, frameCount, prefix = 'Progress:', suffix = 'Complete', length = 50)
            if(cont >= frameCount):
                break
        return np.array(weights)

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
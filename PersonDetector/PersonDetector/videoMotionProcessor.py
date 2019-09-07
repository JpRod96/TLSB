from videoProcessorI import VideoProcessorI
#import matplotlib.pyplot as plt
import util
import cv2
import numpy as np
from scipy.signal import argrelmax
from scipy.signal import argrelmin
from scipy import ndimage
from os import listdir
from os.path import isfile, join
from edgeDetector import EdgeDetector
from PersonDetector import PersonDetector

class VideoMotionProcessor(VideoProcessorI):
    iterations = 4
    MAX_ALIKE_PERCENTAGE = 31
    NONE = 'None'
    EDGE = 'Edges'
    BLURRY_EDGE = 'Blurry_Edges'
    GRAYSCALE = 'Grayscale'
    AUGMENTED_SIZE = 5

    def __init__(self, final_pic_size, to_combine, frames_nr=0, rotate=False, filter=NONE, aug_processor=None):
        self.picSize = final_pic_size
        self.edgeDetector = EdgeDetector(final_pic_size)
        self.detector = PersonDetector()
        self.combineImages = to_combine
        self.framesNumber = frames_nr
        self.rotateImages = rotate
        self.imageFilter = filter
        self.augmentedProcessor = aug_processor

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
            print("Processing frame number "+ str(counter) +"...")
            fileName = self.directory +"/"+ self.videoName + str(counter)
            counter += 1
            frame = frames[index]
            if self.rotateImages:
                frame = ndimage.rotate(frame, 270)
            try:
                frame = self.detector.detectPersonFromNumpy(frame)
                if(self.picSize > 0):
                    frame = self.applyFilter(frame)
                print("Done.\n")
                cv2.imwrite(fileName + ".jpg", frame)
            except:
                print("Human not found on frame number "+ str(counter - 1))
    
    def combineFrames(self, frames, indexes):
        x=1
        edge_images=[]
        for index in indexes:
            frame = frames[index]
            if self.rotateImages:
                frame = ndimage.rotate(frame, 270)
            print("Processing frame number "+ str(x) +"...")
            try:
                treatedImage = self.detector.detectPersonFromNumpy(frame)
                edge_images.append(self.applyFilter(treatedImage))
                print("Done.\n")
            except:
                print("Human not found on frame number "+ str(x))
            x=x+1
        edge_images = self.satisfyFramesNumber(edge_images)
        file_name = self.videoName + "Edges"
        self.concat_images(edge_images, file_name)
    
    def concat_images(self, images, file_name):
        if self.augmentedProcessor is None:
            if len(images) > 0:
                self.create_strip(images, file_name)
            else:
                print("Unsuccessful process, there's no human on the video clip " + self.videoName + "\n")
        else:
            self.augment_image_strip(images)

    def create_strip(self, image_strip, file_name):
        print("Concatenating all images")
        data = util.combineImages(image_strip)
        util.saveImageToPath(data, file_name, ".jpg", self.directory)
        print("Done.")

    def augment_image_strip(self, images_strip):
        if len(images_strip) > 0:
            print("Augmenting image strip...")

            for x in range(0, self.AUGMENTED_SIZE):
                augmented_strip = []

                for image in images_strip:
                    if len(image.shape) == 3:
                        image = np.expand_dims(image, axis=0)
                    else:
                        image = np.expand_dims(image, axis=0)
                        image = np.expand_dims(image, axis=3)
                        print(image.shape)

                    augmented_image_iterator = self.augmentedProcessor.flow(image, batch_size=1)

                    for augmented_image in augmented_image_iterator:
                        image = np.squeeze(augmented_image, axis=0)
                        break
                    augmented_strip.append(image)

                print("Augmented strip number " + str(x+1) + " generated")
                file_name = self.videoName + "EdgesAugmented" + str(x+1)
                self.create_strip(augmented_strip, file_name)
            print("Saving original strip")
            file_name = self.videoName + "EdgesOriginal"
            self.create_strip(images_strip, file_name)
        else:
            print("Unsuccessful process, there's no human on the video clip " + self.videoName + "\n")

    def applyFilter(self, frame):
        filter_name, kh, kw = self.getFilterToken(self.imageFilter)
        if filter_name == self.EDGE:
            return self.edgeDetector.getImageEdgesFromNumpy(frame, kernelHeight = kh, kernelWidth= kw)
        elif filter_name == self.GRAYSCALE:
            return self.edgeDetector.toGrayscale(frame)
        elif filter_name == self.BLURRY_EDGE:
            return self.edgeDetector.getImageBluryEdgesFromNumpy(frame, kernelHeight = kh, kernelWidth= kw)
        else:
            squarePic = self.edgeDetector.make_square(frame)
            return cv2.resize(squarePic, (self.picSize, self.picSize))
    
    def getFilterToken(self, string):
        tokens = string.split()
        if(len(tokens) >= 3):
            return tokens[0], int(tokens[1]), int(tokens[2])
        else:
            return string, 5, 5
    
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
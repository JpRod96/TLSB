import numpy as np
import cv2
import util
from PersonDetector import PersonDetector
from edgeDetector import EdgeDetector
from videoCutter import VideoCutter

class VideoProcessor:
    FRAMES_NRO=5
    fileExtension = ".jpg"

    def processSavingfileNames(self, videoPath):
        fileName = util.getNameOfFileByPath(videoPath)
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, self.FRAMES_NRO)
        videoCutter.saveFrames(frames)
        detector = PersonDetector()
        edgeDetector = EdgeDetector()

        for x in range(1, self.FRAMES_NRO+1):
            imageName = self.fileName + str(x)
            print(imageName + self.fileExtension)
            treatedImageName = detector.detectPerson(imageName, self.fileExtension)
            edgeDetector.getImageEdges(treatedImageName, self.fileExtension)

    def process(self, videoPath, framesNro):
        fileName = util.getNameOfFileByPath(videoPath)
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, framesNro)
        detector = PersonDetector()
        edgeDetector = EdgeDetector()

        x=1;
        edgeImages=[]
        for frame in frames:
            print("Processing "+ str(x) +" frame...")
            treatedImage = detector.detectPersonFronNumpy(frame)
            edgeImages.append(edgeDetector.getImageEdgesFromNumpy(treatedImage))
            print("Done.\n")
            x=x+1
        
        print("Concatenating all images")
        data = util.combineImages(edgeImages)
        util.saveImage(data, fileName+"Edges", ".jpg")
        print("Done.")
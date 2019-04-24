import numpy as np
import cv2
import util
from PersonDetector import PersonDetector
from edgeDetector import EdgeDetector
from videoCutter import VideoCutter

class VideoProcessor:
    framesNro = None
    detector = None
    edgeDetector = None

    def __init__(self, framesNro, finalPicSize):
        self.framesNro = framesNro
        self.detector = PersonDetector()
        self.edgeDetector = EdgeDetector(finalPicSize)

    def processSavingfileNames(self, videoPath):
        fileExtension = ".jpg"
        fileName = util.getNameOfFileByPath(videoPath)
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, self.framesNro)

        for x in range(1, self.framesNro+1):
            imageName = self.fileName + str(x)
            print(imageName + fileExtension)
            treatedImageName = self.detector.detectPerson(imageName, fileExtension)
            self.edgeDetector.getImageEdges(treatedImageName, fileExtension)

    def process(self, videoPath):
        fileName = util.getLastTokenOfPath(videoPath)[0]
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, self.framesNro)

        x=1;
        edgeImages=[]
        for frame in frames:
            print("Processing frame number"+ str(x) +"...")
            try:
                treatedImage = self.detector.detectPersonFronNumpy(frame)
                edgeImages.append(self.edgeDetector.getImageEdgesFromNumpy(treatedImage))
                print("Done.\n")
            except:
                print("Human not found on frame number "+ str(x))
            x=x+1
        
        print("Concatenating all images")
        directory = util.getPathOfVideoDirectory(videoPath)
        data = util.combineImages(edgeImages)
        util.saveImage(data, fileName+"Edges", ".jpg")
        util.saveImageToPath(data, fileName+"Edges", ".jpg", directory)
        print("Done.")
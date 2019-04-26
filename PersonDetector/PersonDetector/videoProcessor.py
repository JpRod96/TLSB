import numpy as np
import cv2
import util
from PersonDetector import PersonDetector
from edgeDetector import EdgeDetector
from videoCutter import VideoCutter

class VideoProcessor:
    framesNro = None
    detector = None
    rotate = None
    edgeDetector = None

    def __init__(self, framesNro, finalPicSize, rotate):
        self.framesNro = framesNro
        self.rotate = rotate
        self.detector = PersonDetector()
        self.edgeDetector = EdgeDetector(finalPicSize)

    def processSavingfileNames(self, videoPath):
        fileExtension = ".jpg"
        fileName = util.getNameOfFileByPath(videoPath)
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, self.framesNro, self.rotate)

        for x in range(1, self.framesNro+1):
            imageName = self.fileName + str(x)
            print(imageName + fileExtension)
            treatedImageName = self.detector.detectPerson(imageName, fileExtension)
            self.edgeDetector.getImageEdges(treatedImageName, fileExtension)

    def process(self, videoPath):
        fileName = util.getLastTokenOfPath(videoPath)[0]
        videoCutter =  VideoCutter()
        frames = videoCutter.cutVideo(videoPath, self.framesNro, self.rotate)

        x=1;
        edgeImages=[]
        for frame in frames:
            print("Processing frame number "+ str(x) +"...")
            try:
                treatedImage = self.detector.detectPersonFronNumpy(frame)
                edgeImages.append(self.edgeDetector.getImageEdgesFromNumpy(treatedImage))
                print("Done.\n")
            except:
                print("Human not found on frame number "+ str(x))
            x=x+1
        if(len(edgeImages)>0):
            print("Concatenating all images")
            directory = util.getPathOfVideoDirectory(videoPath)
            data = util.combineImages(edgeImages)
            util.saveImage(data, fileName+"Edges", ".jpg")
            util.saveImageToPath(data, fileName+"Edges", ".jpg", directory)
            print("Done.")
        else:
            print("Unsuccesful process, theres no human on the video clip "+ videoPath +"\n")
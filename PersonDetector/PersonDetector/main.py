import numpy as np
import cv2
from PersonDetector import PersonDetector
from edgeDetector import EdgeDetector
from videoCutter import VideoCutter
import util

FRAMES_NRO=5
PHOTO = "photo"
JPG_EXTENSION = ".jpg"

def processSavingPhotos():

    videoCutter =  VideoCutter()
    frames = videoCutter.cutVideo("sample.mp4", FRAMES_NRO)
    videoCutter.saveFrames(frames)
    detector = PersonDetector()
    edgeDetector = EdgeDetector()

    for x in range(1, FRAMES_NRO+1):
        imageName = PHOTO + str(x)
        print(imageName + JPG_EXTENSION)
        treatedImageName = detector.detectPerson(imageName, JPG_EXTENSION)
        edgeDetector.getImageEdges(treatedImageName, JPG_EXTENSION)

def process():
    videoCutter =  VideoCutter()
    frames = videoCutter.cutVideo("sample.mp4", FRAMES_NRO)
    detector = PersonDetector()
    edgeDetector = EdgeDetector()

    x=0;
    edgeImages=[]
    for frame in frames:
        imageName = PHOTO + str(x)
        print(imageName + JPG_EXTENSION)
        treatedImage = detector.detectPersonFronNumpy(frame)
        edgeImages.append(edgeDetector.getImageEdgesFromNumpy(treatedImage))
        x=x+1
    
    data = util.combineImages(edgeImages)
    util.saveImage(data, "finalImage", ".jpg")


process()
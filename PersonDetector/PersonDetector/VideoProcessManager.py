from os import listdir
from os.path import isfile, join
from videoProcessor import VideoProcessor
import util

class VideoProcessManager:

    videoProcessor = None
    MP4_EXTENSION = "mp4"

    def __init__(self, videoProcessorImpl):
        self.videoProcessor = videoProcessorImpl
        #VideoProcessor(framesNro, finalPicSize, rotate, typeOfCut)

    def isGivenFileMP4File(self, file):
        finalToken = util.getLastTokenOfPath(file)
        return finalToken[1] == self.MP4_EXTENSION
    
    def isGivenPathADir(self, path):
        finalToken = util.getLastTokenOfPath(path)
        return len(finalToken) == 1
    
    def processPath(self, path):
        if self.isGivenPathADir(path) :
            videoFiles = self.getVideoFilesFromDirectory(path)
            for videoFile in videoFiles:
                self.videoProcessor.process(path + "/" + videoFile)
        else :
            self.videoProcessor.process(path)

    def getVideoFilesFromDirectory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        videoFiles = list(filter(lambda x: self.isGivenFileMP4File(x) , files))
        return videoFiles
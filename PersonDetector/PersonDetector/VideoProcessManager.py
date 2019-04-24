from os import listdir
from os.path import isfile, join
import util

class VideoProcessManager:

    def isGivenPathAFile(self, path):
        finalToken = util.getLastTokenOfPath(path)
        return len(finalToken)>1
    
    def isGivenPathADir(self, path):
        finalToken = util.getLastTokenOfPath(path)
        return len(finalToken)==1
        

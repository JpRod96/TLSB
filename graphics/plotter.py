from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'D:/desktop/TLSB/PersonDetector/PersonDetector')
import util

class Plotter:
    TXT_EXTENSION = 'txt'

    def isGivenFileTxtFile(self, file):
        finalToken = util.getLastTokenOfPath(file)
        return finalToken[1] == self.TXT_EXTENSION
    
    def isGivenPathADir(self, path):
        finalToken = util.getLastTokenOfPath(path)
        return len(finalToken) == 1
    
    #def plotTxtFromPath(self, path):
        
    
    def plotTxtFile(self, filePath):
        txtFile = open(filePath,"r")
        lines = txtFile.readlines()

        objectLine = lines[3]
        objectLine = objectLine.replace(chr(39), '"')
        history = json.loads(objectLine)

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(history['acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.ylim(0, 1)

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epochs')

        plt.suptitle(lines[2])

        plt.savefig('test.png')


    def getTxtFilesFromDirectory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        txtFiles = list(filter(lambda x: self.isGivenFileTxtFile(x) , files))
        return txtFiles

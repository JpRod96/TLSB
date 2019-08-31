from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import asyncio
import time
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
    
    #def plotTxtsFromPath(self, path):
        
    
    def plotTxtFile(self, filePath):
        txtFile = open(filePath,"r")
        lines = txtFile.readlines()
        txtFile.close()

        self.findTestSegments(lines, "test")
    
    def findTestSegments(self, lines, fileName):
        for index in range(0, len(lines)):
            line = lines[index]
            if("--------" in line):
                testNumber = lines[index + 1].split()[2]
                self.findTestEntries(lines, index + 2, fileName + testNumber)

    def findTestEntries(self, lines, initialIndex, fileName):
        line = lines[initialIndex]
        index = initialIndex
        cont = 1
        while(not ("------------" in line)):
            self.plotEntry(lines, index, fileName + str(cont))
            cont += 1
            index += 3
            if(index < len(lines)):
                line = lines[index]
            else:
                break
    
    def plotEntry(self, lines, initialIndex, name):
        objectLine = lines[initialIndex + 1]
        objectLine = objectLine.replace(chr(39), '"')
        history = json.loads(objectLine)

        fig = plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(history['acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.ylim(0, 1)

        plt.subplot(122)
        plt.plot(history['loss'])
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.ylim(0, 3)

        plt.suptitle(lines[initialIndex])
        plt.savefig(name + ".png")
        plt.close(fig)

    def getTxtFilesFromDirectory(self, path):
        files = [f for f in listdir(path) if isfile(join(path, f))]
        txtFiles = list(filter(lambda x: self.isGivenFileTxtFile(x) , files))
        return txtFiles

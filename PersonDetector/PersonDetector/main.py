from VideoProcessManager import VideoProcessManager
from videoProcessor import VideoProcessor
import util

framesNro = 5
picSize = 300
rotate = True
# typeOfCut = "Constant" || "Probabilistic"
typeOfCut = "Constant"

videoProcessor = VideoProcessor(framesNro, picSize, rotate, typeOfCut)
vid = VideoProcessManager(videoProcessor)
vid.processPath("D:/desktop/DATASET/ADIOS")
#vid.processPath("D:/VideosDataset/BANO1")
"""
i = 1
print("Generating edges of images for word Bano")
while i < 6:
  vid.processPath("D:/VideosDataset/BANO" + str(i))
  i += 1
print("Word Bathroom complete")

i = 1
print("Generating edges of images for word BuenosDias")
while i < 6:
  vid.processPath("D:/VideosDataset/BUENOSDIAS" + str(i))
  i += 1
print("Word BuenosDias complete")

i = 1
print("Generating edges of images for word Hola")
while i < 6:
  vid.processPath("D:/VideosDataset/HOLA" + str(i))
  i += 1
print("Word Hola complete")

i = 1
print("Generating edges of images for word Luz")
while i < 6:
  vid.processPath("D:/VideosDataset/LUZ" + str(i))
  i += 1
print("Word Luz complete")

"""
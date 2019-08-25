from VideoProcessManager import VideoProcessManager
from videoCutterProcessor import VideoCutterProcessor
from videoMotionProcessor import VideoMotionProcessor
import util

framesNro = 5
picSize = 500
rotate = True
# typeOfCut = "Constant" || "Probabilistic"
typeOfCut = "Constant"
combineImages = False
filter = 'Blury Edges' #'None' 'Edges' 'Grayscale' Blury Edges'

#videoProcessor = VideoCutterProcessor(framesNro, picSize, rotate, typeOfCut)
videoProcessor = VideoMotionProcessor(picSize, combineImages, filter=filter)

#vid = VideoProcessManager(videoProcessor)
#vid.processPath("D:/desktop/DATASET/CBBA")

videoProcessor.rotateImages = True
videoProcessor.imageFilter = 'Edges'

vid = VideoProcessManager(videoProcessor)
vid.processPath("D:/desktop/DATASET/CBBA/rotar")

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
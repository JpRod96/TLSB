from VideoProcessManager import VideoProcessManager
import util

framesNro = 5
picSize = 300
rotate = True
# typeOfCut = "Constant" || "Probabilistic"
typeOfCut = "Constant"

vid = VideoProcessManager(framesNro, picSize, rotate, typeOfCut)
#vid.processPath("C:/Users/Jp.SANDRO-HP/Downloads/gestos/luz")
#vid.processPath("D:/VideosDataset/BANO1")

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


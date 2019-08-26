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
filter = 'Blury_Edges 1 1' # usando kernel de 1x1
"""
  tipos de filtros
  'None' - ningun filtro, devuelve las imagenes detetadas en rgb y el tamaño de imagen pedido
  'Edges kH kW' - usa detection de bordes por blur guasiano, kH y kW son los tamaños que tendra el kernel del blur 
                  ejemplo: 'Edges 2 4', usara un kernel 2x4, sino se proporcionan los tamaños, se usara un kernel de 5x5
                  ejemplo: 'Edges', usara un kernel de 5x5 
                  Al final se devuelve un imagen en escala de grises con los bordes detectados
  'Grayscale' - usa el filtro de escala de grises y devuelve una imagen con el tamaño pedido
  'Blury_Edges' - usa detection de bordes por blur de Opencv, kH y kW son los tamaños que tendra el kernel del blur 
                  ejemplo: 'Blury_Edges 2 4', usara un kernel 2x4, sino se proporcionan los tamaños, se usara un kernel de 5x5
                  ejemplo: 'Blury_Edges', usara un kernel de 5x5 
                  Al final se devuelve un imagen en escala de grises con los bordes detectados
"""

#videoProcessor = VideoCutterProcessor(framesNro, picSize, rotate, typeOfCut)
videoProcessor = VideoMotionProcessor(picSize, combineImages, filter=filter)

vid = VideoProcessManager(videoProcessor)
vid.processPath("D:/desktop/DATASET/CBBA")

videoProcessor.rotateImages = True
videoProcessor.imageFilter = 'Edges'

vid = VideoProcessManager(videoProcessor)
vid.processPath("D:/desktop/DATASET/CBBA/rotar/VID_20190728_144053.mp4")

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
from VideoProcessManager import VideoProcessManager
from keras.preprocessing.image import ImageDataGenerator
from videoCutterProcessor import VideoCutterProcessor
from videoMotionProcessor import VideoMotionProcessor
from imageProcessor import ImageProcessor
import util

framesNro = 5
picSize = 500  # si es 0 el procesado devuelve imagenes sin ningun filtro ni preprocesamiento
rotate = True
# typeOfCut = "Constant" || "Probabilistic"
typeOfCut = "Constant"
combineImages = True
img_filter = 'Blurry_Edges 4 4'  # usando kernel de 1x1
"""
  tipos de filtros
  'None' - ningun filtro, devuelve las imagenes detetadas en rgb y el tamaño de imagen pedido
  'Edges kH kW' - usa detection de bordes por blur guasiano, kH y kW son los tamaños que tendra el kernel del blur 
                  ejemplo: 'Edges 2 4', usara un kernel 2x4, sino se proporcionan los tamaños, se usara un kernel de 5x5
                  ejemplo: 'Edges', usara un kernel de 5x5 
                  Al final se devuelve un imagen en escala de grises con los bordes detectados
  'Grayscale' - usa el filtro de escala de grises y devuelve una imagen con el tamaño pedido
  'Blurry_Edges' - usa detection de bordes por blur de Opencv, kH y kW son los tamaños que tendra el kernel del blur 
                  ejemplo: 'Blury_Edges 2 4', usara un kernel 2x4, sino se proporcionan los tamaños, se usara un kernel de 5x5
                  ejemplo: 'Blury_Edges', usara un kernel de 5x5 
                  Al final se devuelve un imagen en escala de grises con los bordes detectados
"""
HOLA = "HOLA"
AUTO = "AUTO"
CAFE = "CAFE"
ADIOS = "ADIOS"
GRACIAS = "GRACIAS"
CBBA = "CBBA"
CUAL = "CUAL"
POR_FAVOR = "POR_FAVOR"
QUERER = "QUERER"
YO = "YO"
path = "D:/desktop/DATASET/"
folders = [GRACIAS, CUAL, POR_FAVOR, QUERER]

# videoProcessor = VideoCutterProcessor(framesNro, picSize, rotate, typeOfCut) #procesador de corte sin heuristica
aug = ImageDataGenerator(
    rotation_range=3,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    fill_mode="nearest") #procesador de aumento de datos para inyectar al procesador de video

for folder in folders:
    videoProcessor = VideoMotionProcessor(picSize, combineImages, img_filter=img_filter,
                                          frames_nr=framesNro, aug_processor=aug)  # consultar el constructor de la clase
    vid = VideoProcessManager(videoProcessor)
    vid.processPath(path + folder)
#rotando y cambiando el filtro a edges para la carpeta que yo tengo en el que los videos estan en portrait (verticales/parados)
    videoProcessor.rotateImages = rotate
    videoProcessor.imageFilter = 'Edges'

    vid = VideoProcessManager(videoProcessor)
    vid.processPath(path + folder + "/rotar")#los videos en portrait estan en una carpeta rotar dentro de cada una de las carpetas, esto dentro de mi dataset local
"""
#ejemplo Image processor

imageProcessor = ImageProcessor(picSize)
imageProcessor.blurredEdgeImagesOf("D:/desktop/DATASET/CBBA", 3, 3)
"""

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

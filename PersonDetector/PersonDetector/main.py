from VideoProcessManager import VideoProcessManager
from keras.preprocessing.image import ImageDataGenerator
from videoCutterProcessor import VideoCutterProcessor
from videoMotionProcessor import VideoMotionProcessor
from imageProcessor import ImageProcessor
import cv2
from edgeDetector import EdgeDetector
from HaarCascadeProcessor import HaarCascadeProcessor


def process_image(pic_size):
    # ejemplo Image processor
    path = "C:/Users/Jp/Desktop/Gestos/CAFE/20190805_161744"
    aug = ImageDataGenerator(
        rotation_range=3,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        fill_mode="nearest")
    image_processor = ImageProcessor(pic_size, path)
    # image_processor.augment_images_from(aug, 10)
    image_processor.get_strip_from(image_filter=2, strip_length=5, aug=(aug, 4))


def process_video_motion(folders, pic_size, combine_images, img_filter, frames_nro, path):
    aug = ImageDataGenerator(
        rotation_range=3,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.10,
        fill_mode="nearest")  # procesador de aumento de datos para inyectar al procesador de video, solo se usa al generar tiras

    video_processor = VideoMotionProcessor(pic_size, combine_images, img_filter=img_filter,
                                           frames_nr=frames_nro,
                                           aug_processor=aug)  # consultar el constructor de la clase
    vid = VideoProcessManager(video_processor)
    for folder in folders:
        vid.processPath(path + folder)
        # rotando y cambiando el filtro a edges para la carpeta que yo tengo en el que los videos estan en portrait (verticales/parados)
        video_processor.rotateImages = True
        video_processor.imageFilter = 'Edges'
        vid = VideoProcessManager(video_processor)
        vid.processPath(
            path + folder + "/rotar")  # los videos en portrait estan en una carpeta rotar dentro de cada una de las carpetas, esto dentro de mi dataset local


def setup_variables_for_motion_detector():
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

    folders = [CBBA]
    frames_nro = 5
    pic_size = 500
    combine_images = False
    """
    tipos de filtros para img_filter
    'None' - ningun filtro, devuelve las imagenes detectadas en rgb y el tamaño de imagen pedido
    'Edges kH kW' - usa detection de bordes por blur guasiano, kH y kW (estos valores tiene que ser impares) son los tamaños que tendra el kernel del blur 
                  ejemplo: 'Edges 3 5', usara un kernel 3x5, sino se proporcionan los tamaños, se usara un kernel de 5x5
                  ejemplo: 'Edges', usara un kernel de 5x5 
                  Al final se devuelve un imagen en escala de grises con los bordes detectados
    'Grayscale' - usa el filtro de escala de grises y devuelve una imagen con el tamaño pedido
    'Blurry_Edges KH KW' - usa detection de bordes por blur de Opencv, kH y kW son los tamaños que tendra el kernel del blur 
                          ejemplo: 'Blury_Edges 2 4', usara un kernel 2x4, sino se proporcionan los tamaños, se usara un kernel de 5x5
                          ejemplo: 'Blury_Edges', usara un kernel de 5x5 
                          Al final se devuelve un imagen en escala de grises con los bordes detectados
    """
    img_filter = 'Blurry_Edges 4 4'
    path = "D:/desktop/DATASET/"

    process_video_motion(folders, pic_size, combine_images, img_filter, frames_nro, path)


def process_video_cutter():
    frames_nro = 0
    pic_size = 300
    rotate = True
    type_of_cut = "Constant"  # o 'Probabilistic'

    video_processor = VideoCutterProcessor(frames_nro, pic_size, rotate,
                                           type_of_cut)  # procesador de corte sin heuristica
    vid = VideoProcessManager(video_processor)
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

#process_image(500)
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
folders = [HOLA, AUTO, CAFE, ADIOS, GRACIAS, CBBA, CUAL, POR_FAVOR, QUERER, YO]
path = "C:/Users/Jp/Desktop/Gestos/"


for index in range(1, 12):
    in_path = path + CAFE + "/" + str(index)
    haar = HaarCascadeProcessor()
    haar.process_from(in_path)
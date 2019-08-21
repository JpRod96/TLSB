import numpy as np
import cv2

class EdgeDetector:
    pictureSize = None

    def __init__(self, pictureSize):
        self.pictureSize = pictureSize
    
    def getImageEdges(self, imageName, extension):
        # Cargamos la imagen
        original = cv2.imread(imageName + extension)
 
        # Convertimos a escala de grises
        edgeImage = self.getImageEdgesFromNumpy(original)
        cv2.imwrite(imageName + "edges" + extension, edgeImage)

    def getImageEdgesFromNumpy(self, image):
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ## Aplicar suavizado Gaussiano
        gauss = cv2.GaussianBlur(gris, (5 ,5), 0)
        ## Detectamos los bordes con Canny
        canny = cv2.Canny(gauss, 30, 90)
        #squarePic = self.make_square(canny)
        #resizedImg = cv2.resize(squarePic, (self.pictureSize, self.pictureSize))

        #PRUEBA
        #med_val = np.median(image)
        #lower = int(max(0,0.7*med_val))
        #upper = int(min(255,1.3*med_val))
        #blurred_img = cv2.blur(image,(5,5))
        #canny = cv2.Canny(blurred_img, lower, upper)

        #gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        squarePic = self.make_square(canny)
        resizedImg = cv2.resize(squarePic, (self.pictureSize, self.pictureSize))
        return resizedImg

    def make_square(self, im):
        imgHeight, imgWidth = im.shape[:2]
        desired_size = max([imgHeight, imgWidth])

        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format
        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        r = int(im[0][0][0])
        g = int(im[0][0][1])
        b = int(im[0][0][2])

        color = [r, g, b]
        return cv2.copyMakeBorder(im, 
                                    top, 
                                    bottom, 
                                    left, 
                                    right, 
                                    cv2.BORDER_CONSTANT,
                                    value=color)

        ## Buscamos los contornos
        #(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 
        ## Mostramos el número de monedas por consola
        #print("He encontrado {} objetos".format(len(contornos)))
        # 
        #cv2.drawContours(original,contornos,-1,(0,0,255), 2)
        #cv2.imshow("contornos", original)
 

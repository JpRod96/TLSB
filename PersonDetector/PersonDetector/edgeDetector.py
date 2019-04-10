import numpy as np
import cv2

class EdgeDetector:
    
    def getImageEdges(self, imageName, extension):
        # Cargamos la imagen
        original = cv2.imread(imageName + extension)
 
        # Convertimos a escala de grises
        gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Aplicar suavizado Gaussiano
        gauss = cv2.GaussianBlur(gris, (5 ,5), 0)

        # Detectamos los bordes con Canny
        canny = cv2.Canny(gauss, 50, 150)
 
        #cv2.imshow("canny", canny)

        newimg = cv2.resize(canny, (50,50))

        cv2.waitKey(0)
        cv2. imwrite (imageName + "edges" + extension, canny)


    def squaringPicture(self, img):

        #get size
        height, width, channels = img.shape
        print (in_img,height, width, channels)
        # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((x,y,3), np.uint8)
        #
        #This does the job
        #
        square[(y-height)/2:y-(y-height)/2, (x-width)/2:x-(x-width)/2] = img
        cv2.imwrite(out_img,square)
        cv2.imshow("original", img)
        cv2.imshow("black square", square)
        cv2.waitKey(0)

        ## Buscamos los contornos
        #(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 
        ## Mostramos el número de monedas por consola
        #print("He encontrado {} objetos".format(len(contornos)))
        # 
        #cv2.drawContours(original,contornos,-1,(0,0,255), 2)
        #cv2.imshow("contornos", original)
 

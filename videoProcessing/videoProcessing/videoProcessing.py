import numpy as np
import cv2
 
# Cargamos la imagen
original = cv2.imread("persona.jpg")
cv2.imshow("original", original)
 
# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5 ,5), 0)
 
cv2.imshow("suavizado", gauss)

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
 
#cv2.imshow("canny", canny)

#newimg = cv2.resize(canny, (50,50))
#cv2.imshow("newimg", newimg)
print(canny.shape)
rows,cols = canny.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
dst = cv2.warpAffine(canny,M,(cols,rows))
print(dst.shape)
cv2.waitKey(0)
cv2. imwrite ("resizeimg.jpg", dst)



## Buscamos los contornos
#(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 
## Mostramos el número de monedas por consola
#print("He encontrado {} objetos".format(len(contornos)))
# 
#cv2.drawContours(original,contornos,-1,(0,0,255), 2)
#cv2.imshow("contornos", original)
 

import cv2

carrega_algoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('Photos/img_group_03.jpg')

img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = carrega_algoritmo.detectMultiScale(img_grey, scaleFactor=1.05, minNeighbors=3, minSize=(10, 10))

print(faces)

for(x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Faces', img)
cv2.waitKey()
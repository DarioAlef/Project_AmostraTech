import cv2

carrega_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
carrega_olhos= cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

img = cv2.imread('Photos/img_single_02.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = carrega_olhos.detectMultiScale(img_gray)

#laço de repetição para reconhecimento do rosto
for(x, y, w, h) in faces:
    leitor = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    local_olho = img[y:y+h, x:x+w]
    local_olho_gray = cv2.cvtColor(local_olho, cv2.COLOR_BGR2GRAY)
    detectado = carrega_olhos.detectMultiScale(local_olho_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #laço de repetição para reconhcer olhos
    for(ox, oy, ow, oh) in detectado:
        cv2.rectangle(local_olho, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)

cv2.imshow('Detecta face e os olhos', img)
cv2.waitKey()
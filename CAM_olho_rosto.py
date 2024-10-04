import cv2

webcam = cv2.VideoCapture(0)
classificador_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classificador_olhos = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#laço de repetição para a câmera
while True:
    cam, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = classificador_face.detectMultiScale(gray)

    #laço de repetição para reconhecimento do rosto
    for(x, y, w, h) in detecta:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        caught_eyes = frame[y:y+h, x:x+w]
        eyes_gray = cv2.cvtColor(caught_eyes, cv2.COLOR_BGR2GRAY)
        finder_eyes = classificador_olhos.detectMultiScale(eyes_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #laço de repetição para reconhcer olhos
        for(ox, oy, ow, oh) in finder_eyes:
            cv2.rectangle(caught_eyes, (ox, oy), (ox+ow, oy+oh), (0, 255, 0), 2)

        cv2.imshow('Video webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()
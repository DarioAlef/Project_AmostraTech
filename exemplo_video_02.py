import cv2

webcam = cv2.VideoCapture(0)
frame_video_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    cam, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecta = frame_video_face.detectMultiScale(gray)

    for (x, y, w, h) in detecta:
     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

     cv2.imshow('Video webcam', frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

webcam.release()
cv2.destroyAllWindows()
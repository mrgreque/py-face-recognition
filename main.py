import cv2
import os
from time import sleep
import json

xml_haar_cascade = 'haarcascade_frontalface_alt2.xml'
faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

leiture = False
countImg = 1


pastas = [x[0] for x in os.walk('./images/')]


def showText(img, text, offsety=0):

    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 0.7, 2)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    textY = (img.shape[0] + textsize[1]) - offsety
    cv2.putText(img, text, (textX, textY), font, 0.7, (255, 255, 255), 2)


subjectName = input('Digite o Nome do Sujeito:')
os.mkdir(f'images/{subjectName}/')


while not cv2.waitKey(20) & 0xFF == ord('q'):
    ret, frame_color = capture.read()

    gray = cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    showText(
        frame_color, f'Pressione P para Fotografar', 340)
    showText(frame_color, 'Por favor tire algumas fotos do Seu Rosto. Q para Sair', 40)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if cv2.waitKey(20) & 0xFF == ord('p'):
            print('P Apertado')
            if os.path.exists(f'images/{subjectName}/'):
                imgCrop = gray[y:y+h, x:x+w]
                cv2.imwrite(
                    f'images/{subjectName}/{countImg}.jpg', frame_color)
                countImg += 1
            # leiture = True

    cv2.imshow('color', frame_color)

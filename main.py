import cv2
import os
from time import sleep
import json

xml_haar_cascade = 'haarcascade_frontalface_alt2.xml'
faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

leiture = False
countImg = 1


pastas = [x[0] for x in os.walk('./images/')]
numero = int(''.join([x[0] for x in pastas[-1] if x[0].isnumeric()]))+1


subjectName = input('Digite o Nome do Sujeito:')


while not cv2.waitKey(20) & 0xFF == ord('q'):
    ret, frame_color = capture.read()

    gray = cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY)

    faces = faceClassifier.detectMultiScale(gray)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # if leiture == False:
        if cv2.waitKey(20) & 0xFF == ord('p'):
            print('P Apertado')
            if not os.path.exists(f'images/s{numero}/'):
                print('Fazendo Path')

                os.mkdir(f'images/s{numero}/')
                subjects = []

                with open('./subjects.js', 'r') as file:
                    print('Abrindo arquivo')
                    subjects = json.load(file)
                print(subjects)
                subjects.append(subjectName)

                with open('./subjects.js', 'r') as file:
                    print('Salvar arquivo')
                    json.dump(subjects, file,
                              ensure_ascii=False, indent=4)

                for k in range(20):
                    imgCrop = frame_color[y:y+h, x:x+w]
                    cv2.imwrite(
                        f'images/s{numero}/teste{countImg}.jpg', frame_color)
                    countImg += 1
            # leiture = True

    cv2.imshow('color', frame_color)

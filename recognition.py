import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
names = []


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+h, x:x+w], faces[0]


def prepare_training_data(image_dir):

    faces = []
    labels = []
    index = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                image = cv2.imread(path)

                # pil_image = Image.open(path).convert("L")  # grayscale
                # size = (550, 550)
                # final_image = pil_image.resize(size, Image.ANTIALIAS)
                # image_array = np.array(final_image, "uint8")

                #cv2.imshow('Training...', image)
                cv2.waitKey(100)

                face, rect = detect_face(image)

                if face is not None:
                    (x, y, w, h) = rect

                    #cv2.rectangle(face, (x, y), (x+w, y+h), (0, 255, 255), 4)
                    cv2.imshow('Face detectada', face)

                    if face is not None:
                        faces.append(face)
                        labels.append(index)
                        index += 1
                        names.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels, names


print('Preparing data')
faces, labels, names = prepare_training_data('images')
print('Data prepared')

print('Total faces: ', len(faces))
print('Total labels: ', len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


def draw_text(img, text, rect):
    (x, y, w, h) = rect
    cv2.putText(img, text, (x, y-5),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(imagemCrua):
    img = imagemCrua.copy()

    face, rect = detect_face(img)

    if face is None:
        return imagemCrua

    label = face_recognizer.predict(face)
    print(label)
    try:
        if label[1] < 85:
            label_text = names[label[0]]
        else:
            label_text = 'Nao identificado'
    except:
        label_text = 'Nao identificado'

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect)

    return img


capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while not cv2.waitKey(20) & 0xFF == ord('q'):
    ret, frameCam = capture.read()
    #gray = cv2.cvtColor(frameCam, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('color', frameCam)

    predicted_img = predict(frameCam)
    cv2.imshow('Reconhecimento Facial', predicted_img)

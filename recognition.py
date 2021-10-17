import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

subjects = ["", "O loko bicho", "Beckham"]

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+h, x:x+w], faces[0]

def prepare_training_data(path):

    dirs = os.listdir(f'{path}')

    faces = []
    labels = []

    for dir_name in dirs:

        if not dir_name.startswith('s'):
            continue

        label = int(dir_name.replace('s', ''))

        subject_dir_path = path + '/' + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for img_name in subject_images_names:

            if img_name.startswith('.'):
                continue

            image_path = subject_dir_path + '/' + img_name

            image = cv2.imread(image_path)

            cv2.imshow('Training...', image)
            cv2.waitKey(100)

            face, rect = detect_face(image)

            (x,y,w,h) = rect

            cv2.rectangle(face, (x, y), (x+w, y+h), (0,255,255), 4)
            cv2.imshow('Face detectada', face)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

print('Preparing data')
faces, labels = prepare_training_data('images')
print('Data prepared')

print('Total faces: ', len(faces))
print('Total labels: ', len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img = test_img.copy()

    face, rect = detect_face(img)

    label = face_recognizer.predict(face)
    try: 
        label_text = subjects[label[0]]
    except:
        label_text = 'Nao identificado'

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

# capture = cv2.VideoCapture(0) -> em construção
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while not cv2.waitKey(20) & 0xFF == ord('q'):
    # ret, frame_color = capture.read() -> em construção
    # frame = cv2.imread('fausto.jpg') -> descomentar para teste
    frame = cv2.imread('beck.jpg')
    predicted_img = predict(frame)
    cv2.imshow('Reconhecimento Facial', predicted_img)
    
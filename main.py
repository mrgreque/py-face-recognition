import cv2

xml_haar_cascade = 'haarcascade_frontalface_alt2.xml'
faceClassifier = cv2.CascadeClassifier(xml_haar_cascade)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

while not cv2.waitKey(20) & 0xFF == ord('q'):
    ret, frame_color = capture.read()

    gray = cv2.cvtColor(frame_color, cv2.COLOR_RGB2GRAY)

    faces = faceClassifier.detectMultiScale(frame_color)

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x, y), (x+w, y+h), (0,0,255), 2)
        countImg = 0
        if cv2.waitKey(20) & 0xFF == ord('p'):
            imgCrop = frame_color[y:y+h, x:x+w]
            cv2.imwrite(f'teste{countImg}.jpg', imgCrop)
            cv2.destroyAllWindows()

    cv2.imshow('color', frame_color)
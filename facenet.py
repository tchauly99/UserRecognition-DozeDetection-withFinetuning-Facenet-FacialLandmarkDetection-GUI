import cv2
import os
import configure
import numpy as np
from tensorflow.keras.models import load_model
from imutils import paths


def get_embedding(model_, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    yhat = model_.predict(sample)
    return yhat[0]


class_names = os.listdir(configure.USER)
model = load_model(configure.FACENET_PATH)
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)

embeddings = []
for class_name in class_names:
    print(class_name)
    imagePaths = os.path.sep.join([configure.USER, class_name])
    imagePaths = list(paths.list_images(imagePaths))
    image = cv2.imread(imagePaths[0])
    image = np.resize(image, (160, 160, 3))
    embedding = get_embedding(model, image)
    embeddings.append(embedding)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    clone = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # faces = detector.detect_faces(frame)
    if len(faces) == 0:
        continue
    areas = list()
    for k in range(len(faces)):
        bb = faces[k]
        if bb[0] < 0:
            faces[k][0] = 0
        if bb[1] < 0:
            faces[k][1] = 0
        area = bb[2] * bb[3]
        areas.append(area)
    j = np.argmax(areas)
    bounding_box = faces[j]
    x1 = bounding_box[0]
    x2 = bounding_box[0] + bounding_box[2]
    y1 = bounding_box[1]
    y2 = bounding_box[1] + bounding_box[3]
    image = frame[y1:y2, x1:x2]
    image = np.resize(image, (160, 160, 3))
    target = get_embedding(model, image)
    target = np.array(target)
    distances = []
    for embedding in embeddings:
        distance = np.linalg.norm(embedding-target)
        # distance = reference - target
        # distance = distance*distance
        # distance = np.sum(distance, axis=1)
        # distance = np.square(distance)
        # distance = np.sum(distance)/distance.shape
        distances.append(distance)
    index = np.argmin(distances)
    prob = np.min(distances)
    if prob <= 3:
        label = class_names[index]
    else:
        label = 'Unknown'
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    text = "{}: {}%".format(label, prob)
    cv2.putText(clone, text, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.resize(clone, (20, 20))
    cv2.imshow("result", clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

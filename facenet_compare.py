import cv2
import os
import configure
import numpy as np
from tensorflow.keras.models import load_model
from imutils import paths
import facenet
from scipy import spatial


def get_embedding(model_, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    yhat = model_.predict(sample)
    return yhat[0]


class_names = os.listdir(configure.USER)
print(class_names)
model = load_model(configure.FACENET_PATH)
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)
margin = 11
embeddings = []
for class_name in class_names:
    print(class_name)
    imagePaths = os.path.sep.join([configure.USER, class_name])
    imagePaths = list(paths.list_images(imagePaths))
    image = cv2.imread(imagePaths[0])
    image = np.resize(image, (160, 160, 3))
    prewhitened = facenet.prewhiten(image)
    embedding = get_embedding(model, prewhitened)
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
    det = np.zeros(4, dtype=np.int32)
    det[0] = bounding_box[0]
    det[2] = bounding_box[0] + bounding_box[2]
    det[1] = bounding_box[1]
    det[3] = bounding_box[1] + bounding_box[3]
    det = np.squeeze(det[0:4])
    print(det)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, frame.shape[1])
    bb[3] = np.minimum(det[3] + margin / 2, frame.shape[0])
    print(bb)
    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
    aligned = cv2.resize(cropped, (160, 160))
    prewhitened = facenet.prewhiten(aligned)

    y1 = bb[1]
    y2 = bb[3]
    x1 = bb[0]
    x2 = bb[2]

    target = get_embedding(model, prewhitened)
    target = np.array(target)
    distances = []
    for embedding in embeddings:
        distance = 1 - spatial.distance.cosine(embedding, target)
        distances.append(distance)
    index = np.argmax(distances)
    prob = np.max(distances)
    if prob >= 0.7:
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

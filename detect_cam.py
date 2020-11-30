from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import configure

cap = cv2.VideoCapture(0)
# detector = MTCNN()
faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)

print("[INFO] loading model and label binarizer...")
model = load_model(configure.MODEL_PATH)
lb = pickle.loads(open(configure.LABEL_PATH, "rb").read())

print("[INFO] classifying...")
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
    cut = frame[det[1]:det[3], det[0]:det[2], :]
    y1 = det[1]
    y2 = det[3]
    x1 = det[0]
    x2 = det[2]
    image = frame[y1:y2, x1:x2]
    image = cv2.resize(image, configure.INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    image = img_to_array(image)
    image = np.array(image, dtype="float32")
    image = image.reshape([1, 224, 224, 3])

    pred = model.predict(image)
    pred_index = np.argmax(pred, axis=1)
    prob = pred[:, pred_index]
    if prob >= 0.8:
        label = lb.classes_[pred_index]
    else:
        label = 'Unknown'
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    text = "{}: {}%".format(label, prob * 100)
    cv2.putText(clone, text, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.resize(clone, (20, 20))
    cv2.imshow("result", clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

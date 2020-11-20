from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
from mtcnn.mtcnn import MTCNN
import configure


cap = cv2.VideoCapture(0)
detector = MTCNN()

print("[INFO] loading model and label binarizer...")
model = load_model(configure.MODEL_PATH)
lb = pickle.loads(open(configure.LABEL_PATH, "rb").read())
print("[INFO] classifying image")
while True:
    ret, frame = cap.read()
    clone = frame.copy()
    face = detector.detect_faces(frame)
    if len(face) == 0:
        continue
    bounding_box = face[0]['box']
    x1 = bounding_box[0]
    x2 = bounding_box[0] + bounding_box[2]
    y1 = bounding_box[1]
    y2 = bounding_box[1] + bounding_box[3]
    image = frame[y1:y2, x1:x2]
    print(image.shape)
    image = cv2.resize(image, configure.INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    image = img_to_array(image)
    image = np.array(image, dtype="float32")
    image = image.reshape([1, 224, 224, 3])
    print(image.shape)
    pred = model.predict(image)
    pred_index = np.argmax(pred, axis=1)
    prob = pred[:, pred_index]
    if prob >= 0.7:
        label = lb.classes_[pred_index]
    else:
        label = 'Unknown'
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    text = "{}: {}%".format(label, prob* 100)
    cv2.putText(clone, text, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("result", clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



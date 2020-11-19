from numpy import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import configure
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.applications import ResNet50

clip_path = "clip/Chau.mp4"
cap = cv2.VideoCapture(clip_path)
# detector = MTCNN()
faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)
model = load_model(configure.FACENET_PATH)


def get_embedding(facenet_model, image_raw):
    image_raw = image_raw.astype('float32')
    mean, std = image_raw.mean(), image_raw.std()
    image_raw = (image_raw - mean) / std
    sample = expand_dims(image_raw, axis=0)
    sample = facenet_model.predict(sample)
    return sample[0]


print("[INFO] loading model and label binarizer...")
# model = load_model(configure.MODEL_PATH)
# lb = pickle.loads(open(configure.LABEL_PATH, "rb").read())

with open(configure.SVM_PATH, 'rb') as infile:
    model1, lb = pickle.load(infile)

# with open("output/your_model.pkl", 'rb') as infile:
#     model1, lb = pickle.load(infile)
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
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # faces = detector.detect_faces(frame)
    if len(faces) == 0:
        continue
    # bounding_box = faces[0]['box']
    bounding_box = faces[0]
    x1 = bounding_box[0]
    x2 = bounding_box[0] + bounding_box[2]
    y1 = bounding_box[1]
    y2 = bounding_box[1] + bounding_box[3]
    image = frame[y1:y2, x1:x2]
    image = np.resize(image, (160, 160, 3))
    image = get_embedding(model, image)
    image = expand_dims(image, axis=0)
    pred = model1.predict_proba(image)
    pred_index = np.argmax(pred, axis=1)
    prob = pred[:, pred_index]
    if prob >= 0.5:
        label = lb.classes_[pred_index]
    else:
        label = 'Unknown'
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    text = "{}: {}%".format(label, prob * 100)
    cv2.putText(clone, text, (x1, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.resize(clone, (20, 20))
    cv2.imshow("result", clone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

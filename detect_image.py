from numpy import expand_dims
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import configure
from imutils import paths
import os
from mtcnn.mtcnn import MTCNN

faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)
model = load_model(configure.FACENET_PATH)
detector = MTCNN()

def get_embedding(facenet_model, image_raw):
    image_raw = image_raw.astype('float32')
    mean, std = image_raw.mean(), image_raw.std()
    image_raw = (image_raw - mean) / std
    sample = expand_dims(image_raw, axis=0)
    sample = facenet_model.predict(sample)
    return sample[0]


print("[INFO] loading model and label binarizer...")

with open(configure.SVM_PATH, 'rb') as infile:
    model1, lb = pickle.load(infile)

print("[INFO] classifying...")

path = os.path.sep.join([configure.FROM_CLIP, "Nghia1"])
imagePaths = list(paths.list_images(path))
print("___________________________________________________________")
print(len(imagePaths))
for (i, imagePath) in enumerate(imagePaths):
    print(i)
    image = cv2.imread(imagePath)
    clone = image.copy()
    image = get_embedding(model, image)
    image = expand_dims(image, axis=0)
    pred = model1.predict_proba(image)
    pred_index = np.argmax(pred, axis=1)
    prob = pred[:, pred_index]
    if prob >= 0.5:
        label = lb.classes_[pred_index]
    else:
        label = 'Unknown'
    text = "{}: {}%".format(label, prob * 100)
    cv2.putText(clone, text, (0, 0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.resize(clone, (70, 70))
    cv2.imshow("result", clone)
    cv2.waitKey(0)

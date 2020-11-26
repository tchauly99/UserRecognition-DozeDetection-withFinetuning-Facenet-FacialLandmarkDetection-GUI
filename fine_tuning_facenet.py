from numpy import expand_dims
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import configure
from sklearn.svm import SVC
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
import pandas as pd


def get_embedding(model_, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    sample = np.expand_dims(face, axis=0)
    yhat = model_.predict(sample)
    return yhat[0]


model = load_model(configure.FACENET_PATH)
# print(len(os.listdir(configure.FROM_CLIP)))
# imagePaths = list(paths.list_images(configure.FROM_CLIP))
imagePaths = list(paths.list_images(configure.DATA))
num_class = len(os.listdir(configure.DATA))
print("NUMBER OF CLASS IS {}".format(num_class))

data = list()
labels = list()
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = np.resize(image, (160, 160, 3))
    image = get_embedding(model, image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

labels = np.array(labels)
data = np.array(data, dtype="float32")

# data[data == np.inf] = np.nan
#
# data = pd.DataFrame(data).fillna(0)
# data = np.array(data, dtype="float32")
norm = Normalizer(norm='l2')
data = norm.transform(data)

lb = preprocessing.LabelEncoder()
lb.fit(labels)
labels = lb.transform(labels)

print(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.20, stratify=labels, random_state=42)

model1 = SVC(kernel='linear', probability=True)
model1.fit(trainX, trainY)
yhat_train = model1.predict(trainX)
yhat_test = model1.predict(testX)

score_train = accuracy_score(trainY, yhat_train)
score_test = accuracy_score(testY, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

print("[INFO] saving model...")
with open(configure.SVM_PATH, 'wb') as outfile:
    pickle.dump((model1, lb), outfile)
print("[INFO] saving label encoder...")

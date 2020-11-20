from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
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
import configure
from sklearn.svm import SVC

# imagePaths = list(paths.list_images(configure.FROM_CLIP))
imagePaths = list(paths.list_images(configure.DATA))
data = []
labels = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, configure.INPUT_SIZE)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)

lb = preprocessing.LabelEncoder()
lb.fit(labels)
labels = lb.transform(labels)
# labels = to_categorical(labels)
print(labels.shape)
num_classes = np.max(labels) + 1
print(num_classes)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.10, stratify=labels, random_state=42)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
print("[INFO] preparing model...")

model = ResNet50(weights="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                 include_top=False)

print("[INFO] evaluating network...")
preds = model.predict(data)
preds = preds.reshape(preds.shape[0], -1)
print("[INFO] printing predicts...")
print(preds.shape)
model1 = SVC(kernel='linear', probability=True)
model1.fit(preds, labels)

print("[INFO] saving model...")
# model1.save(configure.MODEL_PATH, save_format="h5")
with open(configure.SVM_PATH, 'wb') as outfile:
    pickle.dump((model1, lb), outfile)
print("[INFO] saving label encoder...")

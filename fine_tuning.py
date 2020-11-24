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
labels = to_categorical(labels)
num_classes = labels.shape[1]
print(num_classes)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.3, stratify=labels, random_state=42)
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")
print("[INFO] preparing model...")

baseModel = ResNet50(weights="imagenet",
                     include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128*2*2, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(num_classes, activation="softmax")(headModel)


# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.25)(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(num_classes, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False
if num_classes == 2:
    loss = "binary_crossentropy"
else:
    loss = "categorical_crossentropy"
opt = Adam(lr=configure.LR)
model.compile(loss=loss, optimizer=opt,
              metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit(aug.flow(trainX, trainY, batch_size=configure.BS),
              steps_per_epoch=len(trainX) // configure.BS,
              validation_data=(testX, testY),
              validation_steps=len(testX) // configure.BS,
              epochs=configure.EPOCHS)

print("[INFO] evaluating network...")
pred = model.predict(testX, batch_size=configure.BS)
pred_index = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1), pred_index, target_names=lb.classes_))

print("[INFO] saving model...")
model.save(configure.MODEL_PATH, save_format="h5")
print("[INFO] saving label encoder...")
f = open(configure.LABEL_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

print("[INFO] printing evaluation result to images")
N = configure.EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(configure.PLOT_PATH)




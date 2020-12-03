from GUI_Form1 import *
from GUI_Test import *
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from imutils import paths
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import dlib
import facenet
import imutils
from imutils import face_utils
import shutil
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import configure
from scipy import spatial
from scipy.spatial import distance as dist
from collections import Counter

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
import matplotlib.pyplot as plt
import pickle


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.timer = QTimer()
        self.timer.timeout.connect(self.viewcam3)

        self.ui.actionOpen_File_2.triggered.connect(self.abdir)
        self.ui.actionOpen_File_2.setShortcut('Ctrl+O')
        self.ui.actionExit_2.triggered.connect(self.exitCall)
        self.ui.actionExit_2.setShortcut('Ctrl+Q')
        print("[INFO] loading HAAR CASCADE...")
        self.face_cascade = cv2.CascadeClassifier(configure.HAAR_PATH)
        self.eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye_tree_eyeglasses.xml')
        print("[INFO] loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(configure.SHAPE_PREDICTOR_PATH)

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0

        self.unlock_counter = 0
        self.labels = []
        self.label = None
        self.ui.tabWidget.currentChanged.connect(self.main_widget_function)
        self.margin = 11

        if not os.path.exists(configure.USER):
            os.makedirs(configure.USER)
        if not os.path.exists(configure.DATA):
            os.makedirs(configure.DATA)
        if not os.path.exists(configure.DATA_RAW):
            os.makedirs(configure.DATA_RAW)
        if not os.path.exists(configure.FROM_CLIP):
            os.makedirs(configure.FROM_CLIP)
        if not os.path.exists(configure.FROM_CLIP_RAW):
            os.makedirs(configure.FROM_CLIP_RAW)
        if not os.path.exists(configure.OUTPUT_PATH):
            os.makedirs(configure.OUTPUT_PATH)

        self.dlib_init_function()
        self.manual_init_function()

        print("[INFO] loading model and label binarizer...")

    def dlib_init_function(self):
        self.mode = 0
        self.model_facenet = load_model(configure.FACENET_PATH)
        self.facenet_compare_setup_function()
        self.class_names = os.listdir(configure.USER)

        self.ui.PlayButton.clicked.connect(self.controlTimer)
        self.ui.PlayButton.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.ui.PlayButton.setEnabled(True)
        self.ui.DeleteUser_Btn.clicked.connect(self.delete_user_function)
        self.ui.Capture_Btn.clicked.connect(self.add_user_function)
        self.ui.Start_Btn.clicked.connect(self.facenet_compare_function)
        self.ui.Stop_Btn.clicked.connect(self.facenet_compare_stop_function)

        self.List_User_function()

    def manual_init_function(self):
        self.num_frame = 0
        self.num_image = 0
        self.mode_2 = 0
        self.output_path = []
        self.class_names_2 = os.listdir(configure.DATA)

        self.ui.Gen_Btn_3.clicked.connect(self.gen_dataset_function)
        self.ui.PlayButton_2.clicked.connect(self.controlTimer)
        self.ui.PlayButton_2.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.ui.PlayButton_2.setEnabled(True)

        self.ui.DeleteUser_Btn_3.clicked.connect(self.delete_user_function)
        self.ui.Capture_Btn_3.clicked.connect(self.add_user_function_2)
        self.ui.Train_Btn.clicked.connect(self.train_function)
        self.ui.Start_Btn_2.clicked.connect(self.compare_function_2)
        self.ui.Stop_Btn_2.clicked.connect(self.facenet_compare_stop_function_2)

        self.List_User_function_2()

    def detect_align_function(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) != 0:
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
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, image.shape[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, image.shape[0])
            aligned = image[bb[1]:bb[3], bb[0]:bb[2], :]
            return aligned, 1, bb
        else:
            aligned = image
            return aligned, 0, 0

    def detect_function(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) != 0:
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
            cut = image[det[1]:det[3], det[0]:det[2], :]
            return cut, 1, det
        else:
            cut = image
            return cut, 0, 0

    def blinking_function(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(self.image, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(self.image, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                if self.COUNTER >= 100:
                    cv2.putText(self.image, "ALERT: DRIVER IS SLEEPING", (200, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                self.COUNTER = 0
            cv2.putText(self.image, "Blinks: {}".format(self.TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 59, 59), 2)
            cv2.putText(self.image, "EAR: {:.2f}".format(ear), (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 59, 59), 2)
            cv2.putText(self.image, "USER: {}".format(self.label), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 59, 59), 2)

    def main_widget_function(self):
        self.COUNTER = 0
        self.TOTAL = 0
        self.unlock_counter = 0
        self.labels = []
        self.label = None

        if self.ui.tabWidget.currentIndex() == 0:
            self.mode = 0
            self.ui.Lock_Lb.setText("Locked")
            self.timer.timeout.connect(self.viewcam3)
        else:
            self.mode_2 = 0
            self.ui.Lock_Lb_2.setText("Locked")
            self.timer.timeout.connect(self.viewcam4)

    def List_User_function_2(self):
        self.ui.ListUser_Cb_3.clear()
        self.class_names_2 = os.listdir(configure.DATA)
        [self.ui.ListUser_Cb_3.addItem(member) for member in self.class_names_2]

    def delete_user_function_2(self):
        path = os.path.sep.join([configure.DATA, self.ui.ListUser_Cb_3.currentText()])
        shutil.rmtree(path)
        self.List_User_function_2()

    def add_user_function_2(self):
        self.timer.timeout.connect(self.viewcam4)
        self.num_frame = 0
        self.num_image = 0
        self.output_path = os.path.sep.join([configure.DATA_RAW, self.ui.username_Tb_3.text()])
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.mode_2 = 1
        self.List_User_function_2()

    def gen_dataset_function(self):
        imagePaths_raw = list(paths.list_images(configure.DATA_RAW))
        class_name = self.ui.username_Tb_3.text()
        print("[INFO]: getting dataset from images")
        print("[INFO]: getting dataset for class {}".format(class_name))
        imagePaths = os.path.sep.join([configure.DATA, class_name])
        if not os.path.exists(imagePaths):
            os.makedirs(imagePaths)
        imagePaths_raw = os.path.sep.join([configure.DATA_RAW, class_name])
        imagePaths_raw = list(paths.list_images(imagePaths_raw))
        i = 0
        for imagePath_raw in imagePaths_raw:

            image_raw = cv2.imread(imagePath_raw)
            aligned, face_existence, _ = self.detect_align_function(image_raw)
            if face_existence == 0:
                print("[INFO] no face detected for {}".format(imagePath_raw))
            else:
                filename = "{}.png".format(i)
                imagePath = os.path.sep.join([imagePaths, filename])
                print(imagePath)
                if aligned is not None and imagePath is not None:
                    cv2.imwrite(imagePath, aligned)
                i += 1

    def train_function(self):
        self.timer.stop()
        imagePaths = list(paths.list_images(configure.DATA))
        BS = int(0.23 * len(imagePaths))
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
        if os.path.exists(configure.RESNET50_WEIGHTS_PATH):
            weights = configure.RESNET50_WEIGHTS_PATH
        else:
            weights = 'imagenet'

        baseModel = ResNet50(weights=weights,
                             include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(128 * 7 * 7, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(num_classes, activation="softmax")(headModel)

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
        H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
                      steps_per_epoch=len(trainX) // BS,
                      validation_data=(testX, testY),
                      validation_steps=len(testX) // BS,
                      epochs=configure.EPOCHS)

        print("[INFO] evaluating network...")
        pred = model.predict(testX, batch_size=BS)
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
        self.timer.timeout.connect(self.viewcam4)

    def compare_function_2(self):
        print("[INFO] loading model and label binarizer...")
        self.model = load_model(configure.MODEL_PATH)
        self.lb = pickle.loads(open(configure.LABEL_PATH, "rb").read())
        self.unlock_counter = 0
        self.labels = []
        self.mode_2 = 2


    def secure_function_2(self):
        cut, face_existence, bb = self.detect_function(self.image)
        if face_existence:
            image = cv2.resize(cut, configure.INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
            image = img_to_array(image)
            image = np.array(image, dtype="float32")
            image = image.reshape([1, 224, 224, 3])
            pred = self.model.predict(image)
            pred_index = np.argmax(pred, axis=1)
            prob = pred[:, pred_index]
            if prob >= 0.8:
                label = self.lb.classes_[pred_index][0]
            else:
                label = 'Unknown'
            self.labels.append(label)
            y1 = bb[1]
            y2 = bb[3]
            x1 = bb[0]
            x2 = bb[2]
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (219, 112, 147), 2)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            text = "{}: {}%".format(label, prob * 100)
            cv2.putText(self.image, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
            self.unlock_counter += 1

    def facenet_compare_stop_function_2(self):
        self.mode_2 = 0
        self.ui.Lock_Lb_2.setText("Locked")
        self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
        # self.timer.stop()

    def viewcam4(self):
        ret, self.image = self.cap.read()

        if self.image is None:
            self.mode = 0
            self.ui.Lock_Lb_2.setText("Locked")
            self.ui.PlayButton_2.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
        else:
            self.image = imutils.resize(self.image, width=400)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if self.mode_2 == 1:
                self.num_frame += 1
                if self.num_frame % 3 == 0:
                    self.num_image += 1
                file_name = "{}.png".format(self.num_image)
                imagePath = os.path.sep.join([self.output_path, file_name])
                if self.image is not None and imagePath is not None:
                    cv2.imwrite(imagePath, self.image)
                if self.num_image == 50:
                    self.mode_2 = 0
                text = "Image number {}".format(self.num_image)
                cv2.putText(self.image, text, (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            elif self.mode_2 == 2:
                self.secure_function_2()

                if self.unlock_counter >= 30:
                    self.unlock_counter = 0
                    print(self.labels)
                    counter = Counter(self.labels)
                    max_value = max(counter.values())
                    print(max_value)
                    max_key = [k for k, v in counter.items() if v == max_value]
                    self.labels = []
                    if (max_key[0] != 'Unknown') and (max_value >= 17):
                        self.label = max_key
                        self.ui.Lock_Lb_2.setText("Unlocked")
                        self.COUNTER = 0
                        self.TOTAL = 0
                        self.mode_2 = 3
            elif self.mode_2 == 3:
                self.blinking_function()

            self.showImage()
            self.checkFPS()

    def List_User_function(self):
        self.ui.ListUser_Cb.clear()
        self.class_names = os.listdir(configure.USER)
        [self.ui.ListUser_Cb.addItem(member) for member in self.class_names]

    def add_user_function(self):
        self.copy = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        user_name = self.ui.username_Tb.text()
        filename = "{}.png".format(user_name)
        path = os.path.sep.join([configure.USER, user_name])
        if not os.path.exists(path):
            os.makedirs(path)
        self.List_User_function()
        imagePath = os.path.sep.join([path, filename])
        aligned, face_existence, _ = self.detect_align_function(self.copy)
        if face_existence == 1:
            cv2.imwrite(imagePath, aligned)

    def delete_user_function(self):
        path = os.path.sep.join([configure.USER, self.ui.ListUser_Cb.currentText()])
        shutil.rmtree(path)
        self.List_User_function()

    def get_embedding(self, model_, face):
        face = face.astype('float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        sample = np.expand_dims(face, axis=0)
        yhat = model_.predict(sample)
        return yhat[0]

    def facenet_compare_function(self):
        self.class_names = os.listdir(configure.USER)
        self.embeddings = []
        for class_name in self.class_names:
            imagePaths = os.path.sep.join([configure.USER, class_name])
            imagePaths = list(paths.list_images(imagePaths))
            image = cv2.imread(imagePaths[0])
            # image = np.resize(image, (160, 160, 3))
            pre_whitened = facenet.prewhiten(image)
            embedding = self.get_embedding(self.model_facenet, pre_whitened)
            self.embeddings.append(embedding)
        self.unlock_counter = 0
        self.labels = []
        self.mode = 1


    def secure_function(self):
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        cut, face_existence, bb = self.detect_function(frame)
        if face_existence:
            pre_whitened = facenet.prewhiten(cut)
            target = self.get_embedding(self.model_facenet, pre_whitened)
            target = np.array(target)
            distances = []
            for embedding in self.embeddings:
                distance = 1 - spatial.distance.cosine(embedding, target)
                distances.append(distance)
            index = np.argmax(distances)
            prob = np.max(distances)
            if prob >= 0.7:
                label = self.class_names[index]
            else:
                label = 'Unknown'
            self.labels.append(label)
            y1 = bb[1]
            y2 = bb[3]
            x1 = bb[0]
            x2 = bb[2]
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (219, 112, 147), 2)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            text = "{}: {}%".format(label, prob)
            cv2.putText(self.image, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    def facenet_compare_setup_function(self):
        self.class_names = os.listdir(configure.USER)
        self.embeddings = []
        for class_name in self.class_names:
            imagePaths = os.path.sep.join([configure.USER, class_name])
            imagePaths = list(paths.list_images(imagePaths))
            image = cv2.imread(imagePaths[0])
            # image = np.resize(image, (160, 160, 3))
            pre_whitened = facenet.prewhiten(image)
            embedding = self.get_embedding(self.model_facenet, pre_whitened)
            self.embeddings.append(embedding)

    def facenet_compare_stop_function(self):
        self.mode = 0
        self.ui.Lock_Lb.setText("Locked")
        self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
        # self.timer.stop()

    def viewcam3(self):
        ret, self.image = self.cap.read()

        if self.image is None:
            self.mode = 0
            self.ui.Lock_Lb.setText("Locked")
            self.TOTAL = 0
            self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stop()
        else:
            self.image = imutils.resize(self.image, width=400)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            if self.mode == 1:
                self.secure_function()
                self.unlock_counter += 1
                if self.unlock_counter >= 30:
                    self.unlock_counter = 0
                    # print(self.labels)
                    counter = Counter(self.labels)
                    max_value = max(counter.values())
                    max_key = [k for k, v in counter.items() if v == max_value]
                    self.labels = []
                    if (max_key[0] != 'Unknown') and (max_value >= 17):
                        self.label = max_key
                        self.ui.Lock_Lb.setText("Unlocked")
                        self.COUNTER = 0
                        self.TOTAL = 0
                        self.mode = 2

            elif self.mode == 2:
                self.blinking_function()

            self.showImage()

            self.checkFPS()

    def showImage(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channel = self.image.shape
        step = channel * width

        qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
        if (self.ui.tabWidget.currentIndex() == 0):
            self.ui.Player.setPixmap(QPixmap.fromImage(qImg))
        else:
            self.ui.Player_2.setPixmap(QPixmap.fromImage(qImg))

    def abdir(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select")

        if filename != "":
            self.ui.PlayButton.setEnabled(True)
            self.filename = filename

    def exitCall(self):
        sys.exit(app.exec_())

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def checkFPS(self):
        self.time1 = datetime.now().microsecond
        deltime = self.time1 - self.time0
        # if deltime != 0:
        #     print(int(round(1000000 / deltime)))
        # self.time0 = self.time1

    def controlTimer(self):
        self.time0 = datetime.now().microsecond

        if not self.timer.isActive():
            clip_path = "clip/Chau.mp4"
            self.cap = cv2.VideoCapture(0)
            # self.cap = cv2.VideoCapture(self.filename)
            self.timer.start(40)
            self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.timer.stop()
            self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
        if (self.ui.tabWidget.currentIndex() == 0):
            self.timer.timeout.connect(self.viewcam3)
        else:
            self.timer.timeout.connect(self.viewcam4)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

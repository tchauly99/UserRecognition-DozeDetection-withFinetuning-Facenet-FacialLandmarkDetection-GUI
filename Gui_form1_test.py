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
import argparse
import imutils
from imutils import face_utils
import shutil
from datetime import datetime

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import configure
from scipy import spatial
from scipy.spatial import distance as dist
from collections import Counter



class AnotherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnotherWindow()
        self.ui.setupUi(self)

        self.ui.ChangeWindow_Btn.clicked.connect(self.change_window_function)

    def change_window_function(self):
        mainwindow = MainWindow()
        mainwindow.show()






class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.timer = QTimer()
        self.timer.timeout.connect(self.viewcam3)
        self.ui.PlayButton.clicked.connect(self.controlTimer)

        self.ui.PlayButton.setIcon(QApplication.style().standardIcon(QStyle.SP_MediaPlay))
        self.ui.PlayButton.setEnabled(False)

        self.ui.actionOpen_File_2.triggered.connect(self.abdir)
        self.ui.actionOpen_File_2.setShortcut('Ctrl+O')

        self.ui.actionExit_2.triggered.connect(self.exitCall)
        self.ui.actionExit_2.setShortcut('Ctrl+Q')

        self.face_cascade = cv2.CascadeClassifier(configure.HAAR_PATH)
        self.eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye_tree_eyeglasses.xml')

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(configure.SHAPE_PREDICTOR_PATH)


        print("[INFO] loading model and label binarizer...")

        print("[INFO] classifying...")

        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        # initialize the frame counters and the total number of blinks
        self.COUNTER = 0
        self.TOTAL = 0

        self.mode = 0

        self.ui.Capture_Btn.clicked.connect(self.capture_function)
        self.ui.Start_Btn.clicked.connect(self.facenet_compare_function)
        self.ui.Stop_Btn.clicked.connect(self.facenet_compare_stop_function)
        self.model = load_model(configure.FACENET_PATH)

        self.unlock_counter = 0
        self.labels = []
        self.label = None
        self.EYE_AR_THRESH = 0.3
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        print("[INFO] loading facial landmark predictor...")
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.class_names = os.listdir(configure.USER)

        self.List_User_function()

        self.ui.DeleteUser_Btn.clicked.connect(self.delete_user_function)

        self.ui.ChangeWindow_Btn.clicked.connect(self.change_window_function)

    def change_window_function(self):
        mainwindow = AnotherWindow()
        mainwindow.show()


    def delete_user_function(self):
        path = os.path.sep.join([configure.USER, self.ui.ListUser_Cb.currentText()])
        shutil.rmtree(path)
        self.List_User_function()


    def List_User_function(self):
        self.ui.ListUser_Cb.clear()
        self.class_names = os.listdir(configure.USER)
        [self.ui.ListUser_Cb.addItem(member) for member in self.class_names]



    def facenet_compare_stop_function(self):
        self.mode = 0
        self.ui.Lock_Lb.setText("Locked")
        self.TOTAL = 0
        self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))
        self.timer.stop()

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
            print(class_name)
            imagePaths = os.path.sep.join([configure.USER, class_name])
            imagePaths = list(paths.list_images(imagePaths))
            image = cv2.imread(imagePaths[0])
            # image = np.resize(image, (160, 160, 3))
            pre_whitened = facenet.prewhiten(image)
            embedding = self.get_embedding(self.model, pre_whitened)
            self.embeddings.append(embedding)

        self.mode = 1



    def capture_function(self):
        self.copy = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        margin = 11
        user_name = self.ui.username_Tb.text()
        filename = "{}.png".format(user_name)
        path = os.path.sep.join([configure.USER, user_name])
        if not os.path.exists(path):
            os.makedirs(path)
        self.List_User_function()
        imagePath = os.path.sep.join([path, filename])

        gray = cv2.cvtColor(self.copy, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(1, 1),
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
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, self.copy.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, self.copy.shape[0])
            aligned = self.copy[bb[1]:bb[3], bb[0]:bb[2], :]

            cv2.imwrite(imagePath, aligned)



    def controlTimer(self):
        self.time0 = datetime.now().microsecond

        if not self.timer.isActive():
            clip_path = "clip/Chau.mp4"
            self.cap = cv2.VideoCapture(self.filename)
            # self.cap = cv2.VideoCapture(self.filename)
            self.timer.start(30)
            self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.timer.stop()
            self.ui.PlayButton.setIcon(QMainWindow().style().standardIcon(QStyle.SP_MediaPlay))


    def viewcam(self):
        ret, self.image = self.cap.read()


        # self.detection()

        # self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # height, width, channel = self.image.shape
        # step = channel*width
        #
        # qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
        # self.ui.Player.setPixmap(QPixmap.fromImage(qImg))
        self.image = cv2.resize(self.image, (480, 280))

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Applying filter to remove impurities
        gray = cv2.bilateralFilter(gray, 5, 1, 1)

        self.faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        if (len(self.faces) > 0):
            for (x, y, w, h) in self.faces:
                self.image = cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # image = self.image[y:(y+h), x:(x+w)]
                # image = cv2.resize(image, configure.INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
                # image = img_to_array(image)
                # image = np.array(image, dtype="float32")
                # image = image.reshape([1, 224, 224, 3])
                # pred = self.model.predict(image)
                # pred_index = np.argmax(pred, axis=1)
                # prob = pred[:, pred_index]
                # # prob = 0.9
                # # pred_index = 1;
                # if prob >= 0.5:
                #     label = self.lb.classes_[pred_index]
                # else:
                #     label = 'Unknown'
                # cv2.rectangle(self.image, (x, y), ((x+w), (y+h)), (0, 0, 255), 2)
                #
                # text = "{}: {}%".format(label, prob * 100)
                # cv2.putText(self.image, text, (x, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.showImage()
        # check FPS
        self.time1 = datetime.now().microsecond
        deltime = self.time1 - self.time0
        print(int(round(1000000 / deltime)))
        self.time0 = self.time1

    def checkFPS(self):
        self.time1 = datetime.now().microsecond
        deltime = self.time1 - self.time0
        print(int(round(1000000 / deltime)))
        self.time0 = self.time1

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
                if self.unlock_counter >= 40:
                    self.unlock_counter = 0
                    counter = Counter(self.labels)
                    max_value = max(counter.values())
                    max_key = [k for k, v in counter.items() if v == max_value]
                    if (max_key[0] != 'Unknown') and (max_value >= 30):
                        self.mode = 2
                        self.label = max_key
                        self.ui.Lock_Lb.setText("Unlocked")

            elif self.mode == 2:
                self.blinking_function()


            self.showImage()

            self.checkFPS()

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
                if self.COUNTER >= 200:
                    cv2.putText(self.image, "ALERT: DRIVER IS SLEEPING", (100, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                self.COUNTER = 0
            cv2.putText(self.image, "Blinks: {}".format(self.TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.image, "EAR: {:.2f}".format(ear), (250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(self.image, "USER: {}".format(self.label), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def secure_function(self):
        margin = 11
        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(1, 1),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # faces = detector.detect_faces(frame)
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
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, frame.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, frame.shape[0])
            aligned = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            # aligned = cv2.resize(aligned, (160, 160))
            pre_whitened = facenet.prewhiten(aligned)
            target = self.get_embedding(self.model, pre_whitened)
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
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            text = "{}: {}%".format(label, prob)
            cv2.putText(self.image, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def viewcam2(self):

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (320, 180))
        if frame is None:
            self.timer.stop()
        self.image = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(1, 1),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) != 0:
            bounding_box = faces[0]
            x1 = bounding_box[0]
            x2 = bounding_box[0] + bounding_box[2]
            y1 = bounding_box[1]
            y2 = bounding_box[1] + bounding_box[3]
            image = frame[y1:y2, x1:x2]
            image = cv2.resize(image, configure.INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
            image = img_to_array(image)
            image = np.array(image, dtype="float32")
            image = image.reshape([1, 224, 224, 3])
            pred = self.model.predict(image)
            pred_index = np.argmax(pred, axis=1)
            prob = pred[:, pred_index]
            if prob >= 0.7:
                label = self.lb.classes_[pred_index]
            else:
                label = 'Unknown'
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            text = "{}: {}%".format(label, prob * 100)
            cv2.putText(self.image, text, (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.resize(self.image, (50, 50))
            self.showImage()

            # check FPS
            self.time1 = datetime.now().microsecond
            deltime = self.time1 - self.time0
            print(int(round(1000000 / deltime)))
            self.time0 = self.time1

    def showImage(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channel = self.image.shape
        step = channel * width

        qImg = QImage(self.image.data, width, height, step, QImage.Format_RGB888)
        self.ui.Player.setPixmap(QPixmap.fromImage(qImg))

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

    # def detection(self):
    #     self.image = imutils.resize(self.image, width=500)
    #     gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     # detect faces in the grayscale image
    #     rects = self.detector(gray, 1)
    #
    #     for (i, rect) in enumerate(rects):
    #         # determine the facial landmarks for the face region, then
    #         # convert the facial landmark (x, y)-coordinates to a NumPy
    #         # array
    #         shape = self.predictor(gray, rect)
    #         shape = face_utils.shape_to_np(shape)
    #         # convert dlib's rectangle to a OpenCV-style bounding box
    #         # [i.e., (x, y, w, h)], then draw the face bounding box
    #         (x, y, w, h) = face_utils.rect_to_bb(rect)
    #         cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         # show the face number
    #         cv2.putText(self.image, "Face #{}".format(i + 1), (x - 10, y - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         # loop over the (x, y)-coordinates for the facial landmarks
    #         # and draw them on the image
    #         for (x, y) in shape:
    #             cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())



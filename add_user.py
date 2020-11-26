import cv2
import os
import configure
import numpy as np
from imutils import paths
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", required=True, help="user name")
args = vars(ap.parse_args())

faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)
margin = 11
user_name = args["user"]
filename = "{}.png".format(user_name)
path = os.path.sep.join([configure.USER, user_name])
if not os.path.exists(path):
    os.makedirs(path)
imagePath = os.path.sep.join([path, filename])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
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
            bb[2] = np.minimum(det[2] + margin / 2, frame.shape[1])
            bb[3] = np.minimum(det[3] + margin / 2, frame.shape[0])
            aligned = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            break

cv2.imwrite(imagePath, aligned)
cv2.destroyAllWindows()
cap.stop()

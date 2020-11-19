import cv2
import os
from mtcnn.mtcnn import MTCNN
from imutils import paths
import configure
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="path to video clip")
args = vars(ap.parse_args())
cap = cv2.VideoCapture(os.path.sep.join([configure.CLIP, args["video"]]))
class_name = args["video"].split(".")[-2]
output_path = os.path.sep.join([configure.FROM_CLIP_RAW, class_name])
num_frame = 0
num_img = 0
print("[INFO]: extracting images from clip")
while True:
    num_frame += 1
    ret, frame = cap.read()
    if (num_frame % 3) == 0:
        num_img += 1
        file_name = os.path.sep.join([output_path, "{}.png".format(num_img)])
        cv2.imwrite(file_name, frame)
    else:
        continue

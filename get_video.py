import cv2
import os
from mtcnn.mtcnn import MTCNN
from imutils import paths
import configure
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--username", required=True)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
num_frame = 0
num_image = 0
output_path = os.path.sep.join([configure.DATA_RAW, args["username"]])
if not os.path.exists(output_path):
    os.makedirs(output_path)
while True:
    ret, frame = cap.read()
    num_frame += 1
    if num_frame % 5 == 0:
        num_image += 1
    file_name = "{}.png".format(num_image)
    imagePath = os.path.sep.join([output_path, file_name])
    if frame is not None and imagePath is not None:
        cv2.imwrite(imagePath, frame)
    if num_image == 50:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    text = "Image number {}".format(num_image)
    cv2.putText(frame, text, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("image", frame)

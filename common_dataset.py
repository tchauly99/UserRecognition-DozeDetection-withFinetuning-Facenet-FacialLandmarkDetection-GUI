import cv2
import os
from mtcnn.mtcnn import MTCNN
from imutils import paths
import configure
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--videos", required=False, nargs="+", help="path to video clips")
# Check whether to use clip, else use images
ap.add_argument("-ic", "--if_clip", required=False, default=False, help="check whether to use clip")
args = vars(ap.parse_args())
# detector = MTCNN()
faceCascade = cv2.CascadeClassifier(configure.HAAR_PATH)
margin = 11
print("----------------------------------------------------")
print(args["if_clip"])
if args["if_clip"]:
    num_img = 0
    num_test = 0
    print(args["if_clip"])
    for clip in args["videos"]:
        # cap = cv2.VideoCapture(os.path.sep.join([configure.CLIP, args["videos"]]))
        cap = cv2.VideoCapture(os.path.sep.join([configure.CLIP, clip]))
        class_name = clip.split(".")[-2]
        output_path = os.path.sep.join([configure.FROM_CLIP_RAW, class_name])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        num_frame = 0
        print("[INFO]: extracting images from clip for class {}".format(class_name))
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            num_test += 1
        division = int(num_test / 50)
        cap = cv2.VideoCapture(os.path.sep.join([configure.CLIP, clip]))
        while True:
            num_frame += 1
            ret, frame = cap.read()
            if frame is None:
                break
            if (num_frame % division) == 0:
                num_img += 1
                filename = "{}.png".format(num_img)
                imagePath = os.path.sep.join([output_path, filename])
                if frame is not None and imagePath is not None:
                    cv2.imwrite(imagePath, frame)
                if num_img == 50:
                    break
            else:
                continue
    print("[INFO]: getting dataset from clip for classes {}".format(args["videos"]))
    if len(args["videos"]) == 1:
        imagePaths_raw = os.path.sep.join([configure.FROM_CLIP_RAW, args["videos"][0].split(".")[-2]])
        imagePaths_raw = list(paths.list_images(imagePaths_raw))
    else:
        imagePaths_raw = list(paths.list_images(configure.FROM_CLIP_RAW))
    i = 0
    for imagePath_raw in imagePaths_raw:
        image_raw = cv2.imread(imagePath_raw)
        # faces = detector.detect_faces(image_raw)
        gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) != 0:
            i += 1
            bounding_box = faces[0]['box']
            # bounding_box = faces[0]
            x1 = bounding_box[0]
            x2 = bounding_box[0] + bounding_box[2]
            y1 = bounding_box[1]
            y2 = bounding_box[1] + bounding_box[3]
            # cv2.rectangle(image_raw, (x1, y1), (x2, y2), (0, 155, 255), 2)
            image = image_raw[y1:y2, x1:x2]
            class_name = imagePath_raw.split(os.path.sep)[-2]
            output_path = os.path.sep.join([configure.FROM_CLIP, class_name])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename = "{}.png".format(i)
            imagePath = os.path.sep.join([output_path, filename])
            # print(imagePath)
            if image is not None and imagePath is not None:
                cv2.imwrite(imagePath, image)
else:
    imagePaths_raw = list(paths.list_images(configure.DATA_RAW))
    class_names = os.listdir(configure.DATA_RAW)
    # class_names = list(["Chau"])
    print("[INFO]: getting dataset from images")
    for class_name in class_names:
        print("[INFO]: getting dataset for class {}".format(class_name))
        imagePaths = os.path.sep.join([configure.DATA, class_name])
        if not os.path.exists(imagePaths):
            os.makedirs(imagePaths)
        imagePaths_raw = os.path.sep.join([configure.DATA_RAW, class_name])
        imagePaths_raw = list(paths.list_images(imagePaths_raw))
        i = 0
        for imagePath_raw in imagePaths_raw:
            image_raw = cv2.imread(imagePath_raw)
            gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # faces = detector.detect_faces(image_raw)
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
                bb[2] = np.minimum(det[2] + margin / 2, image_raw.shape[1])
                bb[3] = np.minimum(det[3] + margin / 2, image_raw.shape[0])
                aligned = image_raw[bb[1]:bb[3], bb[0]:bb[2], :]
                filename = "{}.png".format(i)
                imagePath = os.path.sep.join([imagePaths, filename])
                print(imagePath)
                if aligned is not None and imagePath is not None:
                    cv2.imwrite(imagePath, aligned)
                i += 1
            else:
                print("[INFO] no face detected for {}".format(imagePath_raw))

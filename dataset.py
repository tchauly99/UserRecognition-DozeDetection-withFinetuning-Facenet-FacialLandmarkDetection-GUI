import cv2
import os
from mtcnn.mtcnn import MTCNN
from imutils import paths
import configure

detector = MTCNN()
imagePaths_raw = list(paths.list_images(configure.DATA_RAW))
i = 0
for imagePath_raw in imagePaths_raw:
    image_raw = cv2.imread(imagePath_raw)
    # cv2.imshow("image", image_raw)
    # cv2.waitKey(0)
    faces = detector.detect_faces(image_raw)
    for face in faces:
        i += 1
        bounding_box = face['box']
        x1 = bounding_box[0]
        x2 = bounding_box[0] + bounding_box[2]
        y1 = bounding_box[1]
        y2 = bounding_box[1] + bounding_box[3]
        # cv2.rectangle(image_raw, (x1, y1), (x2, y2), (0, 155, 255), 2)
        image = image_raw[y1:y2, x1:x2]
        class_name = imagePath_raw.split(os.path.sep)[-2]
        filename = "{}.png".format(i)
        imagePath = os.path.sep.join([configure.DATA, class_name, filename])
        print(imagePath)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        if image is not None and imagePath is not None:
            cv2.imwrite(imagePath, image)




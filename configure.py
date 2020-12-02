import os

OUTPUT_PATH = "output"
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "model.h5"])
LABEL_PATH = os.path.sep.join([OUTPUT_PATH, "label.pickle"])
PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "plot.png"])

DATA_RAW = "dataset_raw"
DATA = "dataset"
FOLDER_RAW = os.path.sep.join([DATA_RAW, "TrucAnh"])
FOLDER_NONE_RAW = os.path.sep.join([DATA_RAW, "None"])

USER_RAW = "users_raw"
USER = "users"

CLIP = "clip"
FROM_CLIP_RAW = "datasetraw_fromclip"
FROM_CLIP = "dataset_fromclip"

FOLDER = os.path.sep.join([DATA, "TrucAnh"])
FOLDER_NONE = os.path.sep.join([DATA, "None"])

INPUT_SIZE = (224, 224)
LR = 1e-4
BS = 20
EPOCHS = 20
OUTPUT_PATH = "output"
MODEL_PATH = os.path.sep.join([OUTPUT_PATH, "model.h5"])
LABEL_PATH = os.path.sep.join([OUTPUT_PATH, "label.pickle"])
PLOT_PATH = os.path.sep.join([OUTPUT_PATH, "plot.png"])
SVM_PATH = os.path.sep.join([OUTPUT_PATH, "model_svm.pkl"])

MODELS_PATH = "models"
HAAR_PATH = os.path.sep.join([MODELS_PATH, "haarcascade_frontalface_default.xml"])
RESNET50_WEIGHTS_PATH = os.path.sep.join([MODELS_PATH, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"])
FACENET_PATH = os.path.sep.join([MODELS_PATH, "facenet_keras.h5"])
SHAPE_PREDICTOR_PATH = os.path.sep.join([MODELS_PATH, "shape_predictor_68_face_landmarks.dat"])



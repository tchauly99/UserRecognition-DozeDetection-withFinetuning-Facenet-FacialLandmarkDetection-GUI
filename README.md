# AI_TGM

_**Description**_: 
 * Face detection uses flexibly between **Haar-Cascade** and **Multi-task Cascaded CNN** (MTCNN) method (library available), only the first one is used in GUI.
 * This repo perform face recognition using 3 approachs:
 	* Use **Facenet** model (facenet_keras.h5) to extract embedding vectors from images, compare those vectors between input image and reference images using **cosine** method to detect face matching. (1) (https://medium.com/analytics-vidhya/introduction-to-facenet-a-unified-embedding-for-face-recognition-and-clustering-dbdac8e6f02, https://medium.com/@athul929/building-a-facial-recognition-system-with-facenet-b9c249c2388a)
 	* Train a **Support Vector Machine** (SVM) classifier (with library from sklearn) on embedding vectors extracted from **Facenet**. (2)
 	* Fine-tune (tensorflow - keras) available model structure **ResNet50** with weights from training on dataset **Imagenet** to adapt the model's classifier to our dataset (disable the top layers and add ours). (https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/, https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/) (3)
 * Detect user's drowness from eye-blinking counting, using **facial landmarks** extracted by **dlib**. (4)(https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/, https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/, https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/)
 * Develop a user interface (GUI) to perform the tasks **1, 3, 4** using **PyQt5**.
 * All the seperate source code files can be executed independently to perform seperate tasks, but they are finally combined into Gui_form_test.py to be executed with ease via GUI.
 
Recommended **Python** version: **3.6**

If any error during installing packages, consider Python version.

Install **pip**

Recommended terminal for Windows users: **Cmder**, **Git Bash**

Download IDE Pycharm Community (recommended):
 * https://www.jetbrains.com/pycharm/download/#section=windows

Download code: 
 * $git clone https://github.com/tchauly99/AI_TGM.git

Create and activate your virtual environment (recommended): geeksforgeeks.org/creating-python-virtual-environment-windows-linux/
 * $pip install virtualenv
 * [open terminal in your %Project path%]
 * $virtualenv AI_TGM_venv
 * $source AI_TGM_venv/bin/activate **(on Linux)** or $AI_TGM_venv\Scripts\activate **(on Windows)**
		
Install packages: 
 * $pip install -r requirements.txt

Refer how to download dlib here (for Windows):
 * https://www.geeksforgeeks.org/how-to-install-cmake-for-windows-in-python/
 * https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f
 * or here (for Linux): https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
 
Refer how to config Qt Python Designer and Qt Python UIC for Pycharm here: 
 * https://developpaper.com/pycharm-qt-designer-pyuic-installation-and-configuration-tutorial-details/

Models to download into folder **models**:

 * resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 (just in case of error, because if not downloaded manually, it will still be auto-downloaded): https://github.com/fchollet/deep-learning-models/releases

 * facenet_keras.h5: https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_ or https://www.kaggle.com/suicaokhoailang/facenet-keras

 * haarcascade_frontalface_default.xml: https://gist.github.com/Learko/8f51e58ac0813cb695f3733926c77f52
 
 * shape_predictor_68_face_landmarks.dat: https://osdn.net/projects/sfnet_dclib/downloads/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/

Deactivate virtual enironment after use:
 * $deactivate

**GUIDELINES for seperate little source code files**:

_**Using facenet embeddings**_:

We need an image of every reference user - Press **"q"** on the keyboard to take a photo at the moment of %user name% into users/%user name%
 * $python add_user.py -u %user name%

Perform face similarity recognition for images from webcam - Press **"q"** on the keyboard to terminate:
 * $python facenet_compare.py 

_**Train SVM on facenet embeddings for face recognition**_:

Train SVM on dataset from folder **dataset** (can be modified to folder **dataset_fromclip** in source code):
 * $python fine_tuning_facenet.py

Perform face recognition for images from webcam - Press **"q"** on the keyboard to terminate:
 *  $python detect_clip2.py

_**Fine tune a model for face recognition**_:

We need a dataset of face-cut and aligned images of every user inside folder **dataset** or **dataset_fromclip**, during the process, folder **dataset_raw** or **datasetraw_fromclip** may contain raw images
Get images of each user from webcam into dataset_raw/%user name%:
 *  $python get_video.py -u %user name%
 
Generate dataset into dataset/%user name% for all images in folder **dataset_raw**:
 *  $python common_dataset.py 

Get dataset from clips in folder **clips** - Generate dataset into dataset_fromclip/%clip name% for every %clip name%:
 *  $python common_dataset.py -u %clip name% %clip name% %clip name%... -ic True

Fine-tune **ResNet50** - **Imagenet** on dataset from folder **dataset** (can be modified to folder **dataset_fromcli**p in source code), output model, label and evaluation plot into folder **output**:
 *  $python fine_tuning.py 

Perform face recognition for images from webcam - Press **"q"** on the keyboard to terminate:
 *  $python detect_cam.py
 
Perform face recognition for images from clip - Press **"q"** on the keyboard to terminate:
 *  $python detect_clip.py  -c %clip path%
 
_**Detect drowness through blinking**_:
 *  $python blinking.py

_**GUILDLINES for GUI**_:

$python Gui_form1_test.py

# Facenet tab
![Facenet](/images/Facenet.png)

 * Press **Play** button to start displaying images captured from webcam, you can **Pause** anytime
 * Input %user name% into the **Text box**.
 * Press **Capture** to take a picture of %user name% which will be saved into **users/%user name%** for reference.
 * All user names existing will be placed in the **Combo box**, select a user name out of them and press **Delete User** to delete it from source.
 * Press **Start** to start the process. If the person in front of the camera is recognized as one of the users for 26 frames out of 40 consecutive frames, drowness detection function will be unlocked (**Unlocked** will replaced **Locked** in the label). 
 * Press **Stop** to terminate the process.
 
# Finetune tab
![Finetune](/images/Finetune.png)

 * Press **Play** button to start displaying images captured from webcam, you can **Pause** anytime.
 * Input %user name% into the **Text box**
 * Press **Capture** to get raw images of %user name% into **dataset_raw/%user name%**.
 * Press **Generate** to generate dataset for %user name% into **dataset/%user name%**.
 * All user names existing will be placed in the **Combo box**, select a user name out of them and press **Delete User** to delete it from source.
 * Press **Train Model** to start fine - tuning the model on created dataset.
 * Press **Start** to start the process.
 * Press **Stop** to terminate the process.

**Dir tree**
<pre>         
.
├── clip
├── dataset
│   ├── Chau
│   └── Nghia
├── dataset_fromclip
│   ├── Chau
│   └── Nghia
├── users
│   ├── Chau
│   └── Nghia
├── dataset_raw
│   ├── Chau
│   └── Nghia
├── datasetraw_fromclip
│   ├── Chau
│   └── Nghia
├── models
│   ├── haarcascade_frontalface_default.xml
│   ├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
│   └── facenet_keras.h5
│   └── shape_predictor_68_face_landmarks.dat
├── output
│   ├── label.pickle
│   ├── model.h5
│   └── plot.png
├── images
│   ├── Facenet.png
│   └── Finetune.png
├── add_user.py
├── common_dataset.py
├── configure.py
├── detect_cam.py
├── detect_clip2.py
├── detect_clip.py
├── detect_user.py
├── facenet.py
├── facenet_compare.py 
├── fine_tuning_facenet.py
├── fine_tuning.py
├── get_video.py
├── GUI_Form1.py
├── GUI_Form1.ui
├── Gui_form1_test.py
├── README.md
├── requirements.txt
</pre>

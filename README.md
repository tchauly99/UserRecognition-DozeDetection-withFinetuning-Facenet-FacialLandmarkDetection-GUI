# AI_TGM
Repo to refer: https://medium.com/@athul929/building-a-facial-recognition-system-with-facenet-b9c249c2388a

Download code: git clone https://github.com/tchauly99/AI_TGM.git

Activate your virtualenv:
	pip install virtualenv
	[cd to your Project path]
	virtualenv AI_TGM_venv
	source AI_TGM_venv/bin/activate (on Linux)
		or AI_TGM_venv\Scripts\activate (on Windows)
		
Install packages: pip install -r requirements.txt

Refer how to download dlib here (for Windows):
 * https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f
 * or here (for Linux): https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/

Refer how to config Qt Python Designer and Qt Python UIC for Pycharm here:
 * https://developpaper.com/pycharm-qt-designer-pyuic-installation-and-configuration-tutorial-details/

Deactivate it after use:
 * deactivate
	
Models to download:

 * resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5: https://github.com/fchollet/deep-learning-models/releases

 * facenet_keras.h5: https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_ or https://www.kaggle.com/suicaokhoailang/facenet-keras

 * haarcascade_frontalface_default.xml: https://gist.github.com/Learko/8f51e58ac0813cb695f3733926c77f52
          
<pre>         
.
├── clip
├── dataset
├── dataset_fromclip
│   ├── Chau
│   └── Nghia
├── users
├── dataset_raw
│   ├── Chau2
│   ├── Duong2
│   └── Nghia2
├── datasetraw_fromclip
│   ├── Chau
│   └── Nghia
├── users_raw
├── output
│   ├── label.pickle
│   ├── model.h5
│   └── plot.png
├── __pycache__
│   └── configure.cpython-36.pyc
├── common_dataset.py
├── configure.py
├── dataset_fromclip.py
├── dataset.py
├── datasetraw_fromclip.py
├── dataset_raw.zip
├── detect_cam.py
├── detect_clip2.py
├── detect_clip.py
├── detect_image.py
├── detect_user.py
├── facenet.py
├── fine_tuning_facenet.py
├── fine_tuning.py
├── fine_tuning_svm_resnet.py
├── fine_tuning_vgg16.py
├── get_video.py
├── haarcascade_frontalface_default.xml
├── README.md
├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
└── test.py
</pre>

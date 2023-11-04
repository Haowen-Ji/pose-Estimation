# Sitting Pose Classifer

This is a sitting pose classifier, which can ditinguish cross-leg and head-forward postures. This classifier has also been applied to edge device with
tensorflowlite framework

## 1. Classifier

The classifier is in the pose_classification.ipynb under /main firectory. It has a detalied guidline about the classifier we trained. There are mainly three parts 
in this notebook. 

The first part is a small demo of the movent. The next part is the dataset preprocession, we have comment this file since it cost hours to train images
to landmarks. Thus, we stored the trained data in train_data.csv, traib_data1.csv, test_data.csv

The second part is the baseline cassifier, where the model was trained, results were shown.

The third part is the baseline + object detection model. This model combines the tfmodel model.tflite under /shoe_detect/tensorflow/training_demo/models/detectMode,
which is trained and stored in the object_detection.ipynb file. We then stored the boxes information of each image in train_boxes.npy and test_boxes.npy, since it also 
takes hours to get the bboxes positions.

For the object detection notebook object_detection.ipynb, we used the tensorflow object detection API to train the shoe detection model. But in our later runnings, the code gives diffrent errors under different envrionment. We succeed to train this model in the first time and save it as model.tflite, which is callable. The following is several error messages:

When we run it on colab, the error message is:
ValueError: Unicode strings with encoding declaration are not supported. Please use bytes input or XML fragments without declaration.
The environment is:
dataclasses-0.6 fire-0.5.0 flatbuffers-22.12.6 keras-2.8.0 llvmlite-0.36.0 neural-structured-learning-1.4.0 numba-0.53.0 packaging-20.9 py-cpuinfo-9.0.0 pybind11-2.10.1 scann-1.2.6 sentencepiece-0.1.97 sounddevice-0.4.5 tensorboard-2.8.0 tensorflow-2.8.4 tensorflow-addons-0.19.0 tensorflow-estimator-2.8.0 tensorflow-model-optimization-0.7.3 tensorflowjs-3.18.0 tf-models-official-2.3.0 tf-slim-1.1.0 tflite-model-maker-0.4.2 tflite-support-0.4.3

When we run it on jupyterLab with a GPU, there is sometimes the same error as running on colab, sometimes the error kills the machine
The environment is:
dataclasses-0.6 fire-0.5.0 flatbuffers-22.12.6 keras-2.8.0 llvmlite-0.36.0 neural-structured-learning-1.4.0 numba-0.53.0 packaging-20.9 py-cpuinfo-9.0.0 pybind11-2.10.1 scann-1.2.6 sentencepiece-0.1.97 sounddevice-0.4.5 tensorboard-2.8.0 tensorflow-2.8.4 tensorflow-addons-0.19.0 tensorflow-estimator-2.8.0 tensorflow-model-optimization-0.7.3 tensorflowjs-3.18.0 tf-models-official-2.3.0 tf-slim-1.1.0 tflite-model-maker-0.4.2 tflite-support-0.4.3

## 2. Classifier application 
    
### Main Files

- The `Classifier application.ipynb` file contains the code for running the pre-trained classifier with the device camera or a pre-recorded video.
- The `pose_classifier.tflite` is our pre-trained pose classifier.
- The `media` folder contains sample images and videos.

### Requirements

| Python        | 3.8.13   |
| ------------- | -------- |
| Matplotlib    | 3.5.1    |
| Numpy         | 1.22.3   |
| Opencv-python | 4.6.0.66 |
| Tensorflow    | 2.10.0   |

### Expected Result

The `Classifier application.ipynb` would produce resulting videos in `.avi` format. The resulting videos contain  the labels of the subject poses. As shown below:

![image-20221216005820619](https://github.com/Haowen-Ji/pose-Estimation/blob/master/main/test_out/cross-leg/4835_flip.jpg?raw=true = 250x)

![image-20221216005824629](https://github.com/Haowen-Ji/pose-Estimation/blob/master/main/test_out/normal/3083_flip.jpg?raw=true)

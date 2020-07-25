# Computer Pointer Controller
Openvino version: 2020.2.120, Python 3.7.3

In this project I used 4 pretrained models provided by Intel to build a pointer controller app.
The steps that I followed are: 

1. Face detection to detect the face from the frame(cam or video)
2. From the result of the first model I used the head posed model to estimate the directions of the head
3. From the first model I took the "cropped face" to pass it on the facial landmark estimation model to detect the eye keypoints etc
4. I used the gaze estimation model to estimate the directions of the eye with the necesseray inputs from the previous models
5. Wrap up the results from the models to feed the mouse with the new positon coords.

This project has many potentials for future applications such us helping control the move of mouse for the people who have motion difficulties etc..

## Project Set Up and Installation
Ubuntu-Linux Instructions:

1)You need to install openvino (Tested with: Openvino 2020.2.120)
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

2)Clone this repository

3)Run the openvino enviroment command
Ubuntu example: source /opt/intel/openvino/bin/setupvars.sh

4)Download the 4 models via the model_downloader from the openvino
https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html#model_downloader_usage

Ubuntu Example: 

./downloader.py --name face-detection-adas-binary-0001

Necessery models:
1. face-detection-adas-binary-0001
2. head-pose-estimation-adas-0001
3. landmarks-regression-retail-0009
4. gaze-estimation-adas-0002

Check the requirements file.

## Demo
To run the app you need to run the main.py file

```
python main.py 
-fd models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 
-fl models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 
-hp models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 
-ga models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 
-s video 
-i /bin/demo.mp4 
-vflag fd hp fl ga 
-d CPU 
```


## Documentation
Flag documentation:
Required flags to run: fd,fl,hp,ga,d,i,s,vflag

```
    "-fd" or "--face_detection_model"   = Path to an xml file with a face detection model.
    "-fl" or "--facial_landmarks_model" = Path to an xml file with a facial landmarks model.
    "-hp" or "--head_pose_model"        = Path to an xml file with a head pose model.
    "-ga" or "--gaze_model"             = Path to an xml file with a gaze model.
    "-i" or "--input_path"              = Input path video.
    "-s" or "--input_source"            = Input source (video or cam)
    "-e" or "--extension"               = Path of your extension.
    "-t" or "--threshold"               = Set your prob threshold.
    "-d" or "--device"                  = Specify your target device: ( CPU - GPU - FPGA - MYRIAD )
                            
    "-vflag" or "--visual_flag"         = Specify your visual (models) for each frame:
                                          Values: fd hp fl ga
                                          fd = face detection, fl = facial landmarks
                                          hp = head pose, ga = gaze
    "-vsave" or "--visual_save"         = Visual save option every 10 frames ('y' or 'n')
```
*Visual examples: check the pics inside src folder 

Models:

Face Detection Model: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

Head Pose Estimation Model: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

Facial Landmarks Detection Model: https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

Gaze Estimation Model: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

Tree:

```
.
├── bin
│   └── demo.mp4
├── models
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── pics
│   ├── facedetection2-fp32-.png
│   ├── facedetection-fp32.png
│   ├── gaze-FP16-INT8.png
│   ├── gaze-FP16.png
│   ├── gaze-FP32.png
│   ├── headpose-FP16.png
│   ├── landmarks-FP16-INT8.png
│   ├── landmarks-fp16.png
│   └── landmarks-fp32.png
├── README.md
├── requirements.txt
└── src
    ├── 10_visual.jpg
    ├── 20_visual.jpg
    ├── 30_visual.jpg
    ├── 40_visual.jpg
    ├── 50_visual.jpg
    ├── base_model.py
    ├── facedetection_model.py
    ├── faciallandmarks_model.py
    ├── gaze_model.py
    ├── headpose_model.py
    ├── input_feeder.py
    ├── main.py
    ├── mouse_controller.py
    └── __pycache__
        ├── base_model.cpython-37.pyc
        ├── facedetection_model.cpython-37.pyc
        ├── faciallandmarks_model.cpython-37.pyc
        ├── gaze_model.cpython-37.pyc
        ├── headpose_model.cpython-37.pyc
        ├── input_feeder.cpython-37.pyc
        └── mouse_controller.cpython-37.pyc
```
19 directories, 52 files

bin folder -> the provided video
models folder -> the model that you need to have in order to run the app (check how to download them above)
requirements.txt -> the necessery libraries
pics -> some screenshots of DL-Benchmark tool from the models
src:
    *_visual.jpg files -> the save pic of the visualazitation (vsave flag)
    base_model.py -> the base class model
    facedetection_model.py -> Face Detection model class for handling the facedetection model
    faciallandmarks_model.py -> Facial landmarks estimation class for handling the landmarks estimation model
    gaze_model.py -> Gaze estimation model class for handling the gaze estimation model
    headpose_model.py -> Head pose model class for handling the head pose estimation model
    input_feeder.py -> The class that can handle the input source (cam or video file)
    mouse_controller.py -> the class that can handle via the pyautogui lib the pointer potitions




## Benchmarks
I used the Intel's DL-Benchmark tool that is included in the openvino toolkit.
(Since I don't have access on the Intel's devcloud yet, to test the app on more edge devices like VPU, XEON-CPU and FGPA devices)
With this tool we can have a good first idea of what we can expect from our available devices ( FPS, Latency etc)


## Results
Tested with 9600K CPU with Intel® UHD Graphics 630 - Intel's DL-Benchmark
FPS / Latency-Inference time on Random Generated Dataset

### Face detection
![facedetection1](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/facedetection-fp32.png)
![facedetection2](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/facedetection2-fp32-.png)

### Gaze
![gaze1](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/gaze-FP16.png)          
![gaze2](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/gaze-FP32.png)           
![gaze3](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/gaze-FP16-INT8.png)      

### Facial landmarks
![landmarks1](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/landmarks-fp32.png)
![landmarks2](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/landmarks-fp16.png)
![landmarks3](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/landmarks-FP16-INT8.png)

### Head Pose
![headpose](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/headpose-FP16.png)

### Plots
![plot1](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/gaze-fps.png)
![plot2](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/gaze-inference.png)
![plot3](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/landmarks-fps.png)
![plot4](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/landmarks-inference.png)
![plot5](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/head-fps.png)
![plot6](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/head-inference.png)
![plot7](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/face-fps.png)
![plot8](https://github.com/geochri/Intel_Edge_AI-Computer_Pointer_controller/blob/master/pics/face-inference.png)






The results of FPS and latency time(ms) will be different on real test such as the provided video.
But we already know from the graphs-stats of DL-Benchmark tool that the CPU is better than IGPU on this project.

After becnhmarking with different precisions and my available devices (CPU and iGPU on same cases) I can clearly see the effect of the model's size ( less MB )
from the different precisions FP32, FP16, FP16-INT8. The trick on the different precisions is the computations precisions on the floating numbers etc.
This is a very good technique on the CV, DL and ML  fields because we can have good effecient models. There is always a trade off on the precision strategy such as the accuracy. When we are using lower precision model the results may be not as good as FP32 or FP16. In our case, FP16-INT8 is not as good as FP16 because in order to have a good balance between accuracy detection and effeciency we need to keep a better precision on the computation parts. The reason is that the first two models are the most crucial parts of this "chain". If the first model can not detect corretly the face we will have problems on the rest of the models. Also we can see that FP16-INT8 does not have good enough differences from the FP16. The only good difference is the model's size. Since we want the right balance of effeciency  I will choose the FP16 models. Another observation on the benchmark is the iGPU performance. I believe that the iGPU could perform a little better with async because of the multi core computations. GPU  can procces more frames per second compared to any other hardware and mainly during FP16 because GPU has multiple core and instruction sets that are specifically optimized to run 16bit floating point operations.

### Edge Cases
From what I see on DL-Benchmark tool maybe the right device for this project, even if we had a FPGA device, is the CPU device.
Also another important notice for this project is the results (prediction-estimation) of the first model (facedetection).
Because if the first model cannot detect correctly, the whole chain will be down. So if we want to use this app with the idea of edge app, we need to think ways to improve this idea.
One example is to crop the frame even smaller to "aim" the face to prevent the case of multiple faces on the crowd, because after the first face that will detect, the chain will continue and the results will not be good if the first detected face has not a good potition in front of the camera.

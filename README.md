# Computer Pointer Controller
Openvino version: 2020.2.120
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

1)You need to install openvino (Tested with: Openvino 2020.1.120)
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
```
*Visual examples: check the pics inside src folder 

Models:

Face Detection Model: https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

Head Pose Estimation Model: https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

Facial Landmarks Detection Model: https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

Gaze Estimation Model: https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html





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


The results of FPS and latency time(ms) will be different on real test such as the provided video.
But we already know from the graphs-stats of DL-Benchmark tool that the CPU is better than IGPU on this project.


### Edge Cases
From what I see on DL-Benchmark tool maybe the right device for this project, even if we had a FPGA device, is the CPU device.
Also another important notice for this project is the results (prediction-estimation) of the first model (facedetection).
Because if the first model cannot detect correctly, the whole chain will be down. So if we want to use this app with the idea of edge app, we need to think ways to improve this idea.
One example is to crop the frame even smaller to "aim" the face to prevent the case of multiple faces on the crowd, because after the first face that will detect, the chain will continue and the results will not be good if the first detected face has not a good potition in front of the camera.

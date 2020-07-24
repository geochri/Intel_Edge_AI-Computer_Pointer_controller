import os
import logging
from facedetection_model import Model_Face
from faciallandmarks_model import Model_Faciallandmark
from headpose_model import Model_HeadPose
from gaze_model import Model_Gaze
from input_feeder import InputFeeder
from mouse_controller import MouseController
from argparse import ArgumentParser
import numpy as np
import cv2


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str,
                        help="Path to an xml file with a face detection model.")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str,
                        help="Path to an xml file with a facial landmarks model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to an xml file with a head pose model.")
    parser.add_argument("-ga", "--gaze_model", required=True,
                        help="Path to an xml file with a gaze model.")
    parser.add_argument("-i", "--input_path", required=False, type=str,
                        help="Input path video.")
    parser.add_argument("-s", "--input_source", required=True, type=str,
                        help="Input source (video or cam)")
    parser.add_argument("-e", "--extension", required=False, type=str, default=None,
                        help="Path of your extension.")
    parser.add_argument("-t", "--threshold", required=False, type=str,
                        help="Set your prob threshold", default=0.60)
    parser.add_argument("-d", "--device", required=True, type=str, default='CPU',
                        help="Specify your target device:"
                             "( CPU - GPU - FPGA - MYRIAD )")
    parser.add_argument("-vflag", "--visual_flag", required=True, nargs='+', default=[],
                        help="Specify your visual (models) for each frame:"
                             "Values: fd hp fl ga"
                             "fd = face detection, fl = facial landmarks"
                             "hp = head pose, ga = gaze")
    return parser


def main():
    args = build_argparser().parse_args()
    visual = args.visual_flag
    log = logging.getLogger()
    input_source = args.input_source
    try:
        video_path = args.input_path
    except Exception as e:
        video_path = None
    feed = None
    if input_source.lower() == 'cam':
        feed = InputFeeder('cam')
    elif input_source.lower() == 'video' and os.path.isfile(video_path):
        feed = InputFeeder('video', video_path)
    else:
        log.error('Wrong input feed. (check the video path).')
        exit(1)
        
    fd = Model_Face(args.face_detection_model, args.device, args.extension)
    hp = Model_HeadPose(args.head_pose_model, args.device, args.extension)
    fl = Model_Faciallandmark(args.facial_landmarks_model, args.device, args.extension) 
    ga = Model_Gaze(args.gaze_model, args.device, args.extension)
    ### You can specify the value of precision and speed directly.
    ##  OR
    ## 'high'(100),'low'(1000),'medium','low-med' - precision
    ## 'fast'(1), 'slow'(10), 'medium', 'slow-med' - speed
#     mouse = MouseController('low-med', 'slow-med')
    mouse = MouseController(500,4)
    
    feed.load_data()
    # load models
    fd.load_model()
    hp.load_model()
    fl.load_model()
    ga.load_model()
    count=0
    for ret, frame in feed.next_batch():
        if not ret:
            break
        count+=1
        if count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))
        key = cv2.waitKey(60)
        frame_cp = frame.copy()
        face, face_position = fd.predict(frame_cp, args.threshold)
        if type(face)==int:
            log.error('Prediction Error: Cant find face.')
            if key==27:
                break
            continue
        face_cp = face.copy()
        hp_output = hp.predict(face_cp)
        left_eye, right_eye, facial = fl.predict(face_cp)
#         print('left',left_eye,'\n','right',right_eye,'\n')
        mouse_coord, gaze_vector = ga.predict(left_eye, right_eye, hp_output)
        
        if (not len(visual)==0):
            visual_frame = frame.copy()
            ### Visual FLAGS
            # face detection
            if 'fd' in visual:
                visual_frame = face
            # Head pose
            if 'hp' in visual:
                cv2.putText(visual_frame, "Yaw: {:.2f} Pitch: {:.2f} Roll: {:.2f}".format(hp_output[0],hp_output[1],hp_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 255, 50), 1)
            # Facial landmarks 
            if 'fl' in visual:
                cv2.rectangle(face, (facial[0][0]-10, facial[0][1]-10), (facial[0][2]+10, facial[0][3]+10), (255, 0,0), 3)
                cv2.rectangle(face, (facial[1][0]-10, facial[1][1]-10), (facial[1][2]+10, facial[1][3]+10), (255, 0,0), 3)
            # Gaze estimation
            if 'ga' in visual:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,255,0), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,50,150), 2)
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,255,0), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,50,150), 2)
                face[facial[0][1]:facial[0][3],facial[0][0]:facial[0][2]] = le
                face[facial[1][1]:facial[1][3],facial[1][0]:facial[1][2]] = re
            cv2.namedWindow('Visualization', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('Visualization', 900,900)    
            cv2.imshow('Visualization', cv2.resize(visual_frame, (500,500)))
#             if count%10==0:
#                 cv2.imwrite(str(count)+'_visual.jpg',visual_frame)
        if count%5==0:
            mouse.move(mouse_coord[0], mouse_coord[1])
        if key==27:
            break
        
    log.error('INFO: Ended!')
    cv2.destroyAllWindows()
    feed.close()
    
    
if __name__ == '__main__':
    main()
        
    
    
    
    
    
        
    
        













































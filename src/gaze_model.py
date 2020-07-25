'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import math
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import cv2
import sys
from base_model import Model


class Model_Gaze(Model):
    '''
    Class for the Gaze Model.
    '''

    def predict(self, left_eye, right_eye, head_pose):
        '''
        This method is meant for running predictions on the input image.
        '''
        print('Gaze predict..')
        try:
            self.input_name = [k for k in self.net.inputs.keys()]#next(iter(self.net.inputs))
            self.input_shape = self.net.inputs[self.input_name[1]].shape
            self.output_name = [k for k in self.net.outputs.keys()]#next(iter(self.net.outputs))
#             self.output_shape = self.net.outputs[self.output_name].shape
            print('Re-Initialise.. completed.')
        except Exception as e:
            raise ValueError('Something is wrong with input and output values..')
        left, right = self.preprocess_input(left_eye, right_eye)

        input_dict = {'head_pose_angles':head_pose, 'left_eye_image':left, 'right_eye_image':right}
#         infer = self.net_exec.start_async(request_id=0, inputs=input_dict)
#         status = infer.wait()
        
#         if status == 0:
#             outputs = infer.outputs[self.output_name]
#             coords, gaze_vector = self.preprocess_output(outputs, head_pose)
        results = self.net_exec.infer(input_dict)
        coords, gaze_vector = self.preprocess_output(results, head_pose)
        return coords, gaze_vector

    def check_model(self):
        '''
        Check - initialise the model
        '''
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def preprocess_input(self, left_eye, right_eye):
        '''
        Blob in the format [BxCxHxW]:
        B - batch size
        C - number of channels
        H - image height
        W - image width
        with the name left_eye_image and the shape [1x3x60x60] Blob in the format [BxCxHxW].
        with the name right_eye_image and the shape [1x3x60x60] Blob in the format [BxCxHxW].
        '''
#         print(self.input_shape)
        left_eye = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        left_eye = left_eye.transpose((2, 0, 1))
        left_eye = left_eye.reshape(1, *left_eye.shape)
#         print(right_eye)                      
        right_eye = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))
        right_eye = right_eye.transpose((2, 0, 1))
        right_eye = right_eye.reshape(1, *right_eye.shape)
                               
        return left_eye, right_eye

    def preprocess_output(self, outputs, head_angle):
        '''
        The net outputs a blob with the shape: [1, 3], containing Cartesian coordinates of gaze direction vector
        '''
        gaze_vector = outputs[self.output_name[0]].tolist()[0]
        angle = head_angle[2]
        cosine = math.cos(angle*math.pi/180.0)
        sine = math.sin(angle*math.pi/180.0)
                               
        x = gaze_vector[0] * cosine + gaze_vector[1] * sine
        y =- gaze_vector[0] * sine + gaze_vector[1] * cosine
        
        return (x, y), gaze_vector
        

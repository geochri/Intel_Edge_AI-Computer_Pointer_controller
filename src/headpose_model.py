'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import cv2
import sys
from base_model import Model


class Model_HeadPose(Model):
    '''
    Class for the Head Pose Estimation Model.
    '''


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.image = image
        print('HeadPose predict..')
        pre_image = self.preprocess_input(self.image)
        input_name = self.input_name
        input_dict = {input_name: pre_image}

        results = self.net_exec.infer(input_dict)
        outputs = self.preprocess_output(results)

        return outputs

    def check_model(self):
        '''
        Check - initialise the model
        '''
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def preprocess_input(self, image):
        '''
        An input image in [1xCxHxW] format.

        B - batch size
        C - number of channels
        H - image height
        W - image width
        '''
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):                    
        '''
        Output layer names in Inference Engine format:

            name: "angle_y_fc", shape: [1, 1] - Estimated yaw (in degrees).
            name: "angle_p_fc", shape: [1, 1] - Estimated pitch (in degrees).
            name: "angle_r_fc", shape: [1, 1] - Estimated roll (in degrees).
        '''
        object_list = []
        print('PreOutput-headpose..')
#         print(outputs)
        object_list.append(outputs['angle_y_fc'].tolist()[0][0])
        object_list.append(outputs['angle_p_fc'].tolist()[0][0])
        object_list.append(outputs['angle_r_fc'].tolist()[0][0])

        return object_list

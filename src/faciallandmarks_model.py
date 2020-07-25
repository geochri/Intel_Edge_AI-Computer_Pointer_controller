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


class Model_Faciallandmark(Model):
    '''
    Class for the Facial landmark Model.
    '''

    def predict(self, image, offset=10):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.image = image.copy()
        print('Facial Landmarks predict..')
        pre_image = self.preprocess_input(self.image)
        input_name = self.input_name
        input_dict = {input_name: pre_image}
        results = self.net_exec.infer(input_dict)
        outputs = self.preprocess_output(results)

        height = self.image.shape[0]
        width = self.image.shape[1]
        outputs = outputs* np.array([width, height, width, height])
        outputs = outputs.astype(np.int32)
            
        left_eyexmin = outputs[0] - offset
        left_eyeymin = outputs[1] - offset
        left_eyexmax = outputs[0] + offset
        left_eyeymax = outputs[1] + offset
            
        right_eyexmin = outputs[2] - offset
        right_eyeymin = outputs[3] - offset
        right_eyexmax = outputs[2] + offset
        right_eyeymax = outputs[3] + offset
            
        left_eye_range = self.image[left_eyeymin:left_eyeymax, left_eyexmin:left_eyexmax]
        right_eye_range = self.image[right_eyeymin:right_eyeymax, right_eyexmin:right_eyexmax]
        facial = [[left_eyexmin,left_eyeymin, left_eyexmax,left_eyeymax],
                  [right_eyexmin,right_eyeymin, right_eyexmax,right_eyeymax]] 
        
        return left_eye_range, right_eye_range, facial

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
        An input image in the format [BxCxHxW], where:

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
        The net outputs a blob with the shape: [1, 10], containing a row-vector of 10 floating point values for five landmarks
        coordinates in the form (x0, y0, x1, y1, ..., x5, y5). All the coordinates are normalized to be in range [0,1].
        Preparing the output for the next model (gaze model).
        '''
        object_list = []
        print('PreOutput-facial_landmarks..')
        # left eye(x,y), right eye(x,y)             
        object_list.append(outputs[self.output_name][0][0].tolist()[0][0])
        object_list.append(outputs[self.output_name][0][1].tolist()[0][0])
        object_list.append(outputs[self.output_name][0][2].tolist()[0][0])                   
        object_list.append(outputs[self.output_name][0][3].tolist()[0][0])                   
        
        return object_list
        

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

class Model_Face(Model):
    '''
    Class for the Face Detection Model.
    '''

    def predict(self, image, thres):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.image = image
        print('Face-detection predict..')
        pre_image = self.preprocess_input(self.image)
        input_name = self.input_name
        input_dict = {input_name: pre_image}
#         infer = self.net_exec.start_async(request_id=0, inputs=input_dict)
#         status = infer.wait()
        face = []
#         if status == 0:
# #             print(infer.outputs)
# #             print(self.output_name)
#             results = infer.outputs[self.output_name]
#             outputs = self.preprocess_output(results, thres)
#             outputs = outputs[0]
#             height = self.image.shape[0]
#             width = self.image.shape[1]
#             outputs = outputs* np.array([width, height, width, height])
#             outputs = outputs.astype(np.int32)
#             face = self.image[outputs[1]:outputs[3], outputs[0]:outputs[2]]
            
        results = self.net_exec.infer(input_dict)
        outputs = self.preprocess_output(results, thres)
        outputs = outputs[0]
        height = self.image.shape[0]
        width = self.image.shape[1]
        outputs = outputs* np.array([width, height, width, height])
        outputs = outputs.astype(np.int32)
        face = self.image[outputs[1]:outputs[3], outputs[0]:outputs[2]]
        return face, outputs
        

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

    def preprocess_output(self, outputs, thres):                    
        '''
        thres = threshold of confidence
        
        The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes. 
        Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max], where:

        image_id - ID of the image in the batch
        label - predicted class ID
        conf - confidence for the predicted class
        (x_min, y_min) - coordinates of the top left bounding box corner
        (x_max, y_max) - coordinates of the bottom right bounding box corner.
        '''
        object_list = []
        print('PreOutput-face_detection..')
        tmp_out = outputs[self.output_name][0][0]                   
        for i in tmp_out:
            conf = i[2] # conf-accuracy of the face
            if conf > thres:
                
                x_min = i[3]
                x_max = i[5]
                y_min = i[4]
                y_max = i[6]
            object_list.append([x_min, y_min,x_max, y_max])
        return object_list
                           

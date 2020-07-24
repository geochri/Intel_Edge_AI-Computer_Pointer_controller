'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import cv2
import sys

class Model_Face:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.extensions = extensions
#         self.check_model()
#         try:
#             self.input_name = next(iter(self.net.inputs))
#             self.input_shape = self.net.inputs[self.input_name].shape
#             self.output_name = next(iter(self.net.outputs))
#             self.output_shape = self.net.outputs[self.output_name].shape
#             print('Initialise.. completed.')
#         except Exception as e:
#             raise ValueError('Something is wrong with input and output values..')
        

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            print('Model is loading...')
            self.core = IECore()
            self.net = self.core.read_network(model=self.model_structure, weights=self.model_weights)
            supported = self.core.query_network(self.net, self.device)
            not_supported = [layer for layer in self.net.layers.keys() if layer not in supported]
            if len(not_supported) != 0 and self.device == 'CPU':
                print('Unsuported', not_supported)
                if not self.extensions == None:
                    print('***Quick fix.\n ~CPU Extension added')
                    self.core.add_extension(self.extensions, device)
                    supported = self.core.query_network(self.net, self.device)
                    not_supported = [layer for layer in self.net.layers.keys() if layer not in supported]
                    if len(not_supported) == 0:
                        print('***Quick fix, Failed.')
                else:
                    print('Check the extension path.')
            self.net_exec = self.core.load_network(network=self.net, device_name=self.device)
        except Exception as e:
            raise('Something is wrong.. ~debug load model~')
        
        try:
            self.input_name = next(iter(self.net.inputs))
            self.input_shape = self.net.inputs[self.input_name].shape
            self.output_name = next(iter(self.net.outputs))
            self.output_shape = self.net.outputs[self.output_name].shape
            print('Initialise.. completed.')
        except Exception as e:
            raise ValueError('Something is wrong with input and output values..')
        

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
                           

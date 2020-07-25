'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore
import numpy as np
import os
import cv2
import sys

class Model:
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
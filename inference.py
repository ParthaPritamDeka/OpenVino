from openvino.inference_engine import IENetwork, IEPlugin
from keras.models import load_model
from time import time
from keras.preprocessing import image 
import cv2
from keras.applications import inception_resnet_v2, resnet50
import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def infer_open_vino(net):

    #Use Open Vino CPU plugin
    plugin= IEPlugin(device="CPU")
    defect_input_blob = next(iter(net.inputs))
    defect_exec_net = plugin.load(network=net)

    #Loading the image into memory , preoroceesing and inferencing 
    img_orig = 'path to image'
    img_orig = cv2.resize(img_orig, (224, 224), interpolation=cv2.INTER_CUBIC)
    img_orig = inception_resnet_v2.preprocess_input(img_orig)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = np.array(img_orig).reshape((3,224,224))
  
    res = list(defect_exec_net.infer(inputs={defect_input_blob: img}).values())[0]
    print (res)
       

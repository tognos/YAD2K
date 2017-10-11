#!/usr/bin/env python3
import platform
print("Running with Python version", platform.python_version())

import os
import numpy as np
import keras
from keras.models import Sequential, load_model

from keras.layers import Dense, Conv2D, ZeroPadding2D, Input, Activation
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Add, Multiply, Average, Maximum
from keras.layers import Flatten

from keras.layers.advanced_activations import LeakyReLU, ELU, ThresholdedReLU
from keras.models import Model
from keras import layers
from collections import OrderedDict
import copy
from PIL import Image
import imghdr

from keras.preprocessing import image
from keras import backend as K

import read_activations as du

def file_name_plus(path_name, name_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + name_extension + extension

def changed_extension(path_name, new_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + new_extension

import argparse
import ast
import sys

parser = argparse.ArgumentParser(description='Convert Keras Models to Forge Metal Models',
                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model',
  choices=['INCEPTION_V3', 'RESNET_50','VGG16','INCEPTION_RESNET_V2', 'MOBILE_NET'])
parser.add_argument('input_image', help="path to an imgage or a .float file")
parser.add_argument('--input_dims',
                  help='width and height of the input image; required when using a .float file without shapes dictionary')
parser.add_argument('--features_dir', default="Features",
                  help='Path to directory to write the features to')
parser.add_argument('--model_path',
                  help='Path to a keras .h5 model to use instead of the default one')
parser.add_argument( '-v', '--verbose', help='be verbose about what the converter is doing',
                action='store_true', default = False)
parser.add_argument( '-d', '--debug', help='be very verbose about what the converter is doing',
                action='store_true', default = False)
parser.add_argument( '-s', '--save_orig_models', help='save also the non-optimized keras models as .h5',
                action='store_true', default = False)

opts = parser.parse_args(sys.argv[1:])

MODEL = opts.model
DEBUG_OUT = opts.verbose or opts.debug
MORE_DEBUG_OUT = opts.debug
MODEL_PATH = opts.model_path
FEATURES_DIR = opts.features_dir
INPUT_IMAGE = opts.input_image
INPUT_DIMS = None

if opts.input_dims is not None:
  dims = ast.literal_eval("("+opts.input_dims+")")
  assert len(dims) == 2, "input_dims must be two integers width,height"
  INPUT_DIMS = dims

CUSTOM_OBJECTS = None

if MODEL=="INCEPTION_V3":
  from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
  model_name = "inception_v3"
  if MODEL_PATH is None:
    model = InceptionV3(weights='imagenet')

if MODEL=="RESNET_50":
  from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
  model_name = "resnet_50"
  if MODEL_PATH is None:
    model = ResNnet50(weights='imagenet')

if MODEL=="VGG16":
  from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
  model_name = "vgg_16"
  if MODEL_PATH is None:
    model = VGG16(weights='imagenet')

if MODEL=="INCEPTION_RESNET_V2":
  from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
  model_name = "inception_resnet_v2"
  if MODEL_PATH is None:
    model = InceptionResNetV2(weights='imagenet')

if MODEL=="MOBILE_NET":
  from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions,\
       relu6, DepthwiseConv2D
  model_name = "mobilenet"
  CUSTOM_OBJECTS = {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}
  if MODEL_PATH is None:
    model = MobileNet(weights='imagenet')

if MODEL_PATH is not None:  
  model = load_model(MODEL_PATH, custom_objects = CUSTOM_OBJECTS)

if DEBUG_OUT:
  model.summary()

def dprint(*args, **kwargs):
  if DEBUG_OUT:
    print(*args, **kwargs)

def ddprint(*args, **kwargs):
  if MORE_DEBUG_OUT:
    print(*args, **kwargs)

def dddprint(*args, **kwargs):
  if False:
    print(*args, **kwargs)

import json
def pretty(the_dict):
  return json.dumps(the_dict,sort_keys=True, indent=4)


#############

def stats(features1, message):
  dprint(message)
  dprint("shape:", features1.shape)
  min_1 = np.min(features1)
  max_1 = np.max(features1)
  avrg_1 = np.sum(features1)/features1.size
  dprint("Values: avrg:",avrg_1,"min:",min_1,"max:", max_1)

def load_input_from_img(path, input_shape):
  dprint("Loading "+path)
  img = image.load_img(path, target_size=input_shape)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  stats(x, "Input before preprocessing:")
  x = preprocess_input(x)
  return x 

def load_input_from_floats(path, input_shape):
  dprint("Loading "+path)
  ddprint("input_shape:",input_shape)
  data = np.fromfile(path, dtype = np.float32)
  data = data.reshape(input_shape)
  return data

def write_np_array(the_array, the_file_path):
  c_bytes = the_array.astype(np.float32).tobytes("C")
  with open(the_file_path, "wb") as f:
    f.write(c_bytes)

def predict_image(model, img_path, output_dir, input_shape=None):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  batch_input_shape = model.layers[0].get_config()['batch_input_shape']
  ddprint("batch_input_shape:",batch_input_shape)
  x = None
  if os.path.splitext(img_path)[1] == ".floats":
    if input_shape == None:
      input_shape = [1, batch_input_shape[1], batch_input_shape[2], batch_input_shape[3]]
      x = load_input_from_floats(img_path, input_shape)
    else:
      x = load_input_from_floats(img_path, [1, input_shape[1], input_shape[0], 3])
  else:
    if input_shape == None:
      input_shape = batch_input_shape[1:3]
    x = load_input_from_img(img_path, input_shape)

  print("Loaded input file '{}'".format(img_path))
  stats(x,"Input to model:")
  
  preds = model.predict(x)

  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('Top 3 predictions: {}'.format(decode_predictions(preds, top=3)[0]))

  model_inputs = model.inputs
  print('Computing activations for all layers (This may take about a minute when running Tensorflow runs on the CPU)')
  activations = du.get_activations(model, x, print_shape_only=True, layer_name=None)
  print("Saving {} activations for model '{}' to directory '{}'".format(len(activations),model_name, output_dir))
  shape_info = {}
  prefix = model_name+"-"
  for name, activation in activations.items():
    file_name = prefix+name+".floats"
    testout_path = os.path.join(output_dir, file_name)
    out_trans = activation
    if len(activation.shape) == 4: 
      out_trans = activation.transpose(1, 2, 3, 0).astype("float32")
    elif len(activation.shape) == 3:
      out_trans = activation.transpose(1, 2, 0).astype("float32")
    elif len(activation.shape) == 2:
      out_trans = activation.transpose(1, 0).astype("float32")
    dprint("Saving features to "+testout_path)
    ddprint("Shape: {} Type: {}".format(out_trans.shape, out_trans.dtype))
    write_np_array(out_trans, testout_path)
    shape_info[file_name] = out_trans.shape

  with open(os.path.join(output_dir, "shapes.json"), "w") as json_file:
    print(pretty(shape_info), file=json_file)

  return preds

def predict_in_dir(models, image_dir):
  for image_file in os.listdir(image_dir): 
    try: 
      image_type = imghdr.what(os.path.join(image_dir, image_file)) 
      if not image_type: 
        continue 
    except IsADirectoryError: 
      continue 
    for model in models:
      predict_image(model, os.path.join(image_dir, image_file))


predict_image(model, INPUT_IMAGE, FEATURES_DIR, INPUT_DIMS)
print("Done.")
'''
if MODEL == "INCEPTION_V3":
  #predict_in_dir([model], "images/classify")
  #predict_image(model, os.path.join("images/classify", "zebra.jpg"))
  #predict_image(model, os.path.join("images/classify", "car_299.jpg"), test_out_dir)
  predict_image(model, "/Users/pavel/Downloads/inception_v3/inception_v3-__Activation_3__.floats", test_out_dir)
elif MODEL == "RESNET_50":
  predict_image(model, "/Users/pavel/Downloads/resnet50/resnet_50-subtract_mean.floats", test_out_dir)
else:
  #predict_image(model, os.path.join("images/classify", "car.jpg"), test_out_dir)
  predict_image(model, os.path.join("images/classify", "/Users/pavel/Downloads/vgg_16/vgg_16-subtract_mean.floats"), test_out_dir)
'''
############




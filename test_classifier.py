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

DEBUG_OUT = True
MORE_DEBUG_OUT = False
USE_ORIG_MODEL = False

def file_name_plus(path_name, name_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + name_extension + extension

def changed_extension(path_name, new_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + new_extension

#MODEL="TINY_YOLO"
#MODEL="YOLO"
#as of Aug 29 2017, this official keras inceptions model is still broken with tf backend
#and gives meaningless predictions
#MODEL="INCEPTION_V3"
#MODEL="RESNET_50"
MODEL="VGG16"

model_path = None
test_out_dir=None

if MODEL == "YOLO":
  model_name = "yolo"
  model_path = "model_data/yolo.h5"
  model = load_model(model_path)

if MODEL == "TINY_YOLO":
  model_name = "tiny_yolo"
  model_path = "model_data/tiny-yolo-voc.h5"
  model = load_model(model_path)

if MODEL=="INCEPTION_V3":
  from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
  model_name = "inception_v3"
  model_path = "model_data/inception_v3.h5"
  model = InceptionV3(weights='imagenet')

if MODEL=="RESNET_50":
  from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
  model_name = "resnet_50"
  if K.floatx() == "float16":
    model_path = "model_data/resnet50_nobn.fp16.h5"
  else:
    model_path = "model_data/resnet50.h5"

if MODEL=="VGG16":
  from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
  model_name = "vgg_16"
  if K.floatx() == "float16":
    model_path = "model_data/vgg_16_nobn.fp16.h5"
  else:
    model_path = "model_data/vgg_16.h5"
  model = VGG16(weights='imagenet')

  
if USE_ORIG_MODEL:
  model = load_model(model_path)
  test_out_dir="test_out_orig"
else:
  if K.floatx() == "float16":
    test_out_dir="test_out_fp16"
    model = load_model(model_path)
  else:
    test_out_dir="test_out_nobn"
    model = load_model(file_name_plus(model_path, "_nobn"))


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

def stats(features1):
  print("shape:", features1.shape)
  min_1 = np.min(features1)
  max_1 = np.max(features1)
  avrg_1 = np.sum(features1)/features1.size
  print("avrg1:",avrg_1,"min1:",min_1,"max1:", max_1)

def load_input_from_img(path, input_shape):
  dprint("Loading "+path)
  img = image.load_img(path, target_size=input_shape)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  stats(x)
  x = preprocess_input(x)
  return x 

def load_input_from_floats(path, input_shape):
  dprint("Loading "+path)
  print("input_shape:",input_shape)
  data = np.fromfile(path, dtype = np.float32)
  data = data.reshape(input_shape)
  return data

def write_np_array(the_array, the_file_path):
  c_bytes = the_array.astype(np.float32).tobytes("C")
  with open(the_file_path, "wb") as f:
    f.write(c_bytes)

def predict_image(model, img_path, output_dir):
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  batch_input_shape = model.layers[0].get_config()['batch_input_shape']
  print("batch_input_shape:",batch_input_shape)
  x = None
  if os.path.splitext(img_path)[1] == ".floats":
    input_shape = [1, batch_input_shape[1], batch_input_shape[2], batch_input_shape[3]]
    x = load_input_from_floats(img_path, input_shape)
  else:
    input_shape = batch_input_shape[1:3]
    x = load_input_from_img(img_path, input_shape)

  stats(x)
  
  preds = model.predict(x)

  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  dprint('Predicted:', decode_predictions(preds, top=3)[0])

  model_inputs = model.inputs
  activations = du.get_activations(model, x, print_shape_only=True, layer_name=None)

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
    print("out_trans: ", out_trans.shape, out_trans.dtype)
    print("Saving features to "+testout_path)
    #out_tests.tofile(testout_path)
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

if MODEL == "INCEPTION_V3" or MODEL=="RESNET_50":
  #predict_in_dir([model], "images/classify")
  #predict_image(model, os.path.join("images/classify", "zebra.jpg"))
  #predict_image(model, "/Users/pavel/Downloads/resnet-50/resnet_50-subtract_mean.floats", "test_out-fp16")
  predict_image(model, "/Users/pavel/Downloads/resnet-50/resnet_50-subtract_mean.floats", test_out_dir)
else:
  #predict_image(model, os.path.join("images/classify", "car.jpg"), test_out_dir)
  predict_image(model, os.path.join("images/classify", "/Users/pavel/Downloads/vgg_16/vgg_16-subtract_mean.floats"), test_out_dir)

############




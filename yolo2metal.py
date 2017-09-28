#!/usr/bin/env python3

# Conversion script for a number of keras models to Metal.
#
# The pretrained YOLOv2 model was made with the Darknet framework. You first
# need to convert it to a Keras model using YAD2K, and then yolo2metal.py can 
# convert the Keras model to Metal.
# 
# Required packages: python, numpy, h5py, pillow, tensorflow, keras.
# if you want to plot graphs: dot, graphviz
#
#
# For yolo, download the tiny-yolo-voc.weights and tiny-yolo-voc.cfg files:
# wget https://pjreddie.com/media/files/tiny-yolo-voc.weights
# wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/tiny-yolo-voc.cfg
#
# Install YAD2K:
# https://github.com/allanzelener/YAD2K/
#
# Run the yad2k.py script to convert the Darknet model to Keras:
# ./yad2k.py -p tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.h5
#
# Edit the model_path variable to point to where tiny-yolo-voc.h5 was saved.
#
# Finally, run yolo2metal.py. It will convert the weights to Metal format
# and save them to the "Parameters" directory.

print("*** keras2metal converter v0.1 ***")

import os
import sys
import datetime
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
from keras.preprocessing import image
from collections import OrderedDict
import copy
from PIL import Image
import imghdr

from keras.preprocessing import image
import argparse

'''
VERBOSE = True
MORE_VERBOSE = False
RUN_CLASSIFIER = True
PLOT_MODELS = True
SAVE_ORIGINAL_MODEL = True
EXPORT_ORIGINAL_WEIGHTS = False

PARAMETERS_OUT_DIR = "Parameters"
PARAMETERS_ORIG_OUT_DIR = "ParametersOrig"
'''

parser = argparse.ArgumentParser(description='Convert Keras Models to Forge Metal Models',
                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model',
  choices=['TINY_YOLO', 'YOLO', 'INCEPTION_V3', 'RESNET_50','VGG16','INCEPTION_RESNET_V2'])
parser.add_argument('--param_dir', default="Parameters",
                  help='Path to directory to write the metal weights for optimized models')
parser.add_argument('--orig_param_dir', default="OriginalParameters",
                  help='Path to directory to write the metal weights of non-optimized models')
parser.add_argument('--model_dir', default='model_data',
                  help='Path to directory where keras and yolo models are stored')
parser.add_argument('--src_out_dir', default='generated_src',
                  help='Path to directory where the swift source code will be written')
parser.add_argument('--image_dir', default='images/classify',
                  help='Path to directory where images for classification tests are to be found')
parser.add_argument( '-p', '--plot_models', help='Plot the models and save as image.',
                action='store_true', default = False)
parser.add_argument( '-v', '--verbose', help='be verbose about what the converter is doing',
                action='store_true', default = False)
parser.add_argument( '-d', '--debug', help='be very verbose about what the converter is doing',
                action='store_true', default = False)
parser.add_argument( '-r', '--run_classifier', help='run a classification models on test input',
                action='store_true', default = False)
parser.add_argument( '-s', '--save_orig_models', help='save also the non-optimized keras models as .h5',
                action='store_true', default = False)
parser.add_argument( '-e', '--export_orig_weigths', help='export also the non-optimized model weigths',
                action='store_true', default = False)
parser.add_argument( '-q', '--quick_run', help='skip optimzed model generation and weight export reusing model from previous run',
                action='store_true', default = False)



opts = parser.parse_args(sys.argv[1:])

MODEL = opts.model
VERBOSE = opts.verbose or opts.debug
MORE_VERBOSE = opts.debug
PLOT_MODELS= opts.plot_models
SAVE_ORIGINAL_MODEL=opts.save_orig_models
RUN_CLASSIFIER=opts.run_classifier
MODEL_DIR=opts.model_dir
EXPORT_ORIGINAL_WEIGHTS=opts.export_orig_weigths
IMAGE_DIR=opts.image_dir
QUICK_RUN=opts.quick_run
SRC_OUT_DIR=opts.src_out_dir
PARAMETERS_OUT_DIR=opts.param_dir
PARAMETERS_ORIG_OUT_DIR=opts.orig_param_dir

print(opts)

if RUN_CLASSIFIER:
  if MODEL == "YOLO" or MODEL == "TINY_YOLO":
    print("WARNINNG: classifier will not be run on YOLO; use test_yolo.py instead")
    RUN_CLASSIFIER = False

SWAP_INPUT_IMAGE_CHANNELS = False # when true, code for input channel swap RGB->BGR will be generated
SUBTRACT_IMAGENET_MEAN = False # when true, code for imagenet mean subtraction will be generated
SCALE_INPUT_TO_MINUS_1_AND_1 = True # when true, the input image values will be scaled to -1 .. 1

if MODEL == "YOLO":
  model_name = "yolo"
  model_path = MODEL_DIR+"/yolo.h5"
  model = load_model(model_path)

if MODEL == "TINY_YOLO":
  model_name = "tiny_yolo"
  model_path = MODEL_DIR+"/tiny-yolo-voc.h5"
  model = load_model(model_path)

if MODEL=="INCEPTION_V3":
  from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
  SCALE_INPUT_TO_MINUS_1_AND_1 = True
  model_name = "inception_v3"
  model_path = MODEL_DIR+"/inception_v3.h5"
  model = InceptionV3(weights='imagenet')
  if SAVE_ORIGINAL_MODEL:
    model.save(model_path)

if MODEL=="INCEPTION_RESNET_V2":
  from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
  SCALE_INPUT_TO_MINUS_1_AND_1 = True
  model_name = "inception_resnet_v2"
  model_path = MODEL_DIR+"/inception_resnet_v2.h5"
  model = InceptionResNetV2(weights='imagenet')
  if SAVE_ORIGINAL_MODEL:
    model.save(model_path)


if MODEL=="RESNET_50":
  from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
  SWAP_INPUT_IMAGE_CHANNELS = True
  SUBTRACT_IMAGENET_MEAN = True
  model_name = "resnet_50"
  model_path = MODEL_DIR+"/resnet50.h5"
  model = ResNet50(weights='imagenet')
  if SAVE_ORIGINAL_MODEL:
    model.save(model_path)

if MODEL=="VGG16":
  from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
  SWAP_INPUT_IMAGE_CHANNELS = True
  SUBTRACT_IMAGENET_MEAN = True
  model_name = "vgg_16"
  model_path = MODEL_DIR+"/vgg_16.h5"
  model = VGG16(weights='imagenet')
  if SAVE_ORIGINAL_MODEL:
    model.save(model_path)

def file_name_plus(path_name, name_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + name_extension + extension

def changed_extension(path_name, new_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + new_extension

if VERBOSE:
  model.summary()

pydot = None

def import_pydot():
  try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
  except ImportError:
    if MORE_VERBOSE:
      print("Failed to import pydot_ng, falling back on pydotplus")
    # pydotplus is an improved version of pydot
    try:
      import pydotplus as pydot
    except ImportError:
      # Fall back on pydot if necessary.
      if MORE_VERBOSE:
        print("Failed to import pydotplus, falling back on pydot")
      try:
        import pydot
      except ImportError:
        print("Failed to import pydot, nothing to fall back on")
        pydot = None
  return pydot
  
def check_pydot():
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
  except Exception:
    # pydot raises a generic Exception here,
    # so no specific class can be caught.
    raise ImportError('Failed to import pydot. You must install pydot'
      ' and graphviz for `pydotprint` to work.')

if PLOT_MODELS:
  pydot = import_pydot()
  check_pydot()
  from keras.utils import plot_model
  plot_model(model, to_file=changed_extension(model_path,'.png'))

def dprint(*args, **kwargs):
  if VERBOSE:
    print(*args, **kwargs)

def ddprint(*args, **kwargs):
  if MORE_VERBOSE:
    print(*args, **kwargs)

# silence output, change to True if you want to have that one, too
def dddprint(*args, **kwargs):
  if False:
    print(*args, **kwargs)


#############
def top_classes(preds, top=5):
  results = []
  confidences = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    top_preds = [pred[i] for i in top_indices]
    results.append(top_indices)
    confidences.append(top_preds)
  return results, confidences

def predict_image(model, img_path):
  batch_input_shape = model.layers[0].get_config()['batch_input_shape']
  input_shape = batch_input_shape[1:3]
  if input_shape[0] == None or input_shape[1] == None:
    input_shape = (299,299)
  print("\nLoading "+img_path)
  img = image.load_img(img_path, target_size=input_shape)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  ddprint(str(x.shape))

  preds = model.predict(x)

  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)
  print('Predicted:', decode_predictions(preds, top=5)[0])
  top_i, top_p = top_classes(preds, top=5)
  print('Top5 Classes:', top_i[0])
  print('Top5 Confid.:', top_p[0])
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

if RUN_CLASSIFIER:
  predict_in_dir([model], IMAGE_DIR)

############


import json
def pretty(the_dict):
  return json.dumps(the_dict,sort_keys=True, indent=4)

def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    dddprint("Folding bn "+bn_layer.__class__.__name__+":\n"+pretty(bn_layer.get_config())+"\ninto conv "+conv_layer.__class__.__name__+":i\n"+pretty(conv_layer.get_config())+"\n")
    conv_weights = conv_layer.get_weights()[0]

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    next_index = 0
    if bn_layer.scale:
      dprint("batch_norm has scale (gamma)")
      gamma = bn_weights[next_index]
      next_index += 1
    else:
      gamma = 1.0
    if bn_layer.center:
      dprint("batch_norm has center (beta)")
      beta = bn_weights[next_index]
      next_index += 1
    else:
      beta = 0

    mean = bn_weights[next_index]
    variance = bn_weights[next_index + 1]

    dprint("bn_weights.shape="+str(np.shape(bn_weights)))
    
    epsilon = float(bn_layer.get_config()['epsilon'])
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    return new_weights, new_bias

all_layers = []
prev_layer = None
prev_prev_layer = None
new_weights = []


# returns a dict mapping all layer names to arrays with the names of their inbound layers
# layers with no inbound layers map to an empty array
def find_inbound_layers(model):
  model_config = model.get_config()
  layers_config = model_config['layers']

  inbound_layer_by_name = {}

  for layer_dict in layers_config:
    layer_name = layer_dict['name']
    inbound_names = []
    inbound_nodes = layer_dict['inbound_nodes']
    if len(inbound_nodes) > 0:
      #print(str(inbound_nodes))
      inbound_node_list = inbound_nodes[0]
      for node in inbound_node_list:
        inbound_names.append(node[0])
      ddprint("Layer name:"+layer_name+": Inbound:"+str(inbound_names))
    else:
      ddprint("Layer name:"+layer_name+": No inbound layers")
    inbound_layer_by_name[layer_name] = inbound_names
  return inbound_layer_by_name


ddprint(json.dumps(model.get_config(),sort_keys=True, indent=4))

#convenience function returning the name of a layer from the config
def layer_name(layer):
  return layer.get_config()['name']

# In Keras the only way to clone a layer (without the weights)
# is to serialize and deserialize the layer config
# This function does that
def layer_clone(layer):
  return layers.deserialize({'class_name': layer.__class__.__name__,
                                    'config': layer.get_config()})

# returns a a replacement for a the_name if it is in the
# replacements dict; otherwise, the_name is returned
# This is used to reference the correct layers
# when they have been optimized away
def replaced_name(the_name, replacements):
  if the_name in replacements:
    return replacements[the_name]
  else:
    return the_name

# returns an array of names that might have been replaced
# by the function above
def replaced_name_list(the_names, replacements):
  result = []
  for name in the_names:
    result.append(replaced_name(name, replacements))
  return result

# return output blobs of inbound layers to layer_by_name
def input_layers_outputs(inbound_names, layer_by_name):
  #print("input_layers_outputs: inbound_names: "+str(inbound_names)+", all="+str(layer_by_name))
  ddprint("input_layers_outputs: inbound_names: "+str(inbound_names))
  if len(inbound_names) == 1:
    result = layer_by_name[inbound_names[0]]
    #print("input_layers_outputs: returning single result: "+str(result))
    return result;
  else:
    result = []
    for name in inbound_names:
      result.append(layer_by_name[name])
    #print("input_layers: returning list result: "+str(result))
    return result

# return layers from model specified by inbound_names 
'''
def orig_input_layers(inbound_names, model):
  if len(inbound_names) == 1:
    result = model.get_layer(name=inbound_names[0])
    #print("orig_input_layers: returning single result: "+str(result))
    return result 
  else:
    result = []
    for iname in inbound_names:
      result.append(model.get_layer(name=iname))
    #print("orig_input_layers: returning single result: "+str(result))
    return result
'''
def orig_input_layers(inbound_names, model):
  result = []
  for iname in inbound_names:
    result.append(model.get_layer(name=iname))
  return result


inbound_by_name = find_inbound_layers(model)

# dict all layers by name for the new model
layer_by_name = {} 

# dict of all weights by layer name for the new model
weights_by_name = {}

# dict of the output blob of a new layer by name
output_by_name = {}

# dict of layer names that have been optimized away
# so that layers referencing these layers
# can reference the output of the remaining layer
replaced_layer = {}

# register a new layer and its output blob in
# above global dicts
def register_new_layer(the_name, the_layer, the_output):
  all_layers.append(the_output)
  layer_by_name[the_name] = the_layer
  output_by_name[the_name] = the_output

if not QUICK_RUN:
  print("Creating optimized model (with batchnorm folding):\n")
  # iterate over all layers of the original model
  for index, layer in enumerate(model.layers):
    #print("\n"+str(index)+":"+layer.__class__.__name__+":"+str(layer.get_config()))
    dprint(str(index)+":"+layer.__class__.__name__+":"+layer_name(layer))

    # get names of the inbound layers of our current layer
    inbounds = replaced_name_list(inbound_by_name[layer_name(layer)], replaced_layer)
    dprint("Inbounds:"+str(inbounds)+", len "+str(len(inbounds)))

    if len(inbounds) == 0:
      # layer has no inbounds, so we assume it is an input layer
      # and create a new input layer
      batch_input_shape = layer.get_config()['batch_input_shape']
      input_shape = batch_input_shape[1:]
      dprint(str(input_shape), str(batch_input_shape))
      new_layer = Input(shape=input_shape)
      # Note: new_layer is actually the blob output of the input
      register_new_layer(layer_name(layer), new_layer, new_layer)
    else:
      # the layer has at least one input
      orig_inputs = orig_input_layers(inbounds, model)
      ddprint("orig_inputs:"+str(orig_inputs))
      ddprint("orig_inputs class:"+str(type(orig_inputs)))
      # check if we need to complete a skipped Conv2D without batchnorm
      if layer.__class__.__name__ != "BatchNormalization":
        for i in orig_inputs:
          if i.__class__.__name__ == "Conv2D" and i.name not in weights_by_name:
            # conv without following batchnorm, set normal weights for previous conv layer
            dprint("Conv2d layer without following batchnorm, set orig weight for layer ",i.name)
            prev_orig_layer = model.get_layer(name=i.name)
            new_layer = layer_clone(prev_orig_layer)
            prev_inbounds = replaced_name_list(inbound_by_name[layer_name(prev_orig_layer)],replaced_layer)
            inputs = input_layers_outputs(prev_inbounds, output_by_name)
            register_new_layer(layer_name(new_layer), new_layer, new_layer(inputs))
            weights_by_name[layer_name(new_layer)]=prev_orig_layer.get_weights()
         
      if not layer.get_weights():
        # layer has no weights, so we just clone it and instantiate
        # and connect it using the keras functional API
        dprint("Layer '"+layer.__class__.__name__+"' has no weights")
        new_layer = layer_clone(layer)
        inputs = input_layers_outputs(inbounds, output_by_name)
        register_new_layer(layer_name(layer), new_layer, new_layer(inputs))
      else:
        # layer has weights, so we might have to do some optimization
        weights_list = layer.get_weights()
        for i, w in enumerate(weights_list):
          dprint("Layer '"+layer.__class__.__name__+"' weights/biases index "+str(i)+" of shape:"+str(w.shape))
        layer_done = False
        #print("orig_inputs:"+str(orig_inputs)+", class "+str(orig_inputs.__class__))
        #print("BNTEST:"+layer.__class__.__name__ +" "+ orig_inputs.__class__.__name__)
        
        if layer.__class__.__name__ == "BatchNormalization" and\
            orig_inputs[0].__class__.__name__ == "Conv2D":
          # batchnorm following a conv2D layer
          # we need to set folded weights for the previous conv layer
          # which also has not been created yet
          dprint("Folding batch norm layer")
          prev_orig_layer = orig_input_layers(inbounds, model)[0]
          new_config = prev_orig_layer.get_config()
          # we need our new conv layer to have bias
          new_config['use_bias'] = True
          # create a conv layer
          new_layer = layers.deserialize({'class_name': prev_orig_layer.__class__.__name__,
                                    'config': new_config})
          prev_inbounds = replaced_name_list(inbound_by_name[layer_name(prev_orig_layer)],replaced_layer)
          inputs = input_layers_outputs(prev_inbounds, output_by_name)
          register_new_layer(layer_name(new_layer), new_layer, new_layer(inputs))
          dprint("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
          weights_by_name[layer_name(new_layer)] = fold_batch_norm(prev_orig_layer, layer)
          # add the name of the new layer as a replacement for the folded batchnorm layer
          replaced_layer[layer_name(layer)] = layer_name(prev_orig_layer)
          layer_done = True
        if not layer_done:
          # process all layer types except conv2d if not the last layer
          if layer.__class__.__name__ != "Conv2D" or index + 1 == len(model.layers):
            new_layer = layer_clone(layer)
            inputs = input_layers_outputs(inbounds, output_by_name)
            register_new_layer(layer_name(layer), new_layer, new_layer(inputs))
            dprint("appending new layer:"+new_layer.__class__.__name__)
            dprint("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
            weights_by_name[layer_name(layer)] = layer.get_weights()

  # create our new model using the keras functional API
  new_model = Model(inputs=all_layers[0], outputs=all_layers[-1])
else:
  print("Loading optimized model (with batchnorm folding)\n")
  new_model = load_model(file_name_plus(model_path, "_nobn"))

if VERBOSE:
  new_model.summary()
if PLOT_MODELS:
  plot_model(new_model, to_file=changed_extension(file_name_plus(model_path,'_new'),'.png'))

if not QUICK_RUN:
  ddprint("Replaced layers:"+str(replaced_layer))

  # now actually set the weights for the new model layers
  # we have to do it after model instantiation because
  # the keras could not calculate the actual shapes
  # before the model was completely set up
  print("Setting weights for new model")
  for layer_name_, weights in weights_by_name.items():
    dprint("Setting weights for layer "+layer_name_+" type " + layer_by_name[layer_name_].__class__.__name__)
    #print("weights     :"+str(np.shape(weights)))
    #print("orig_weights:"+str(np.shape(layer_by_name[layer_name_].get_weights())))
    layer_by_name[layer_name_].set_weights(weights)

if RUN_CLASSIFIER:
  predict_in_dir([new_model], IMAGE_DIR)


# for some unknow reasons, ndarray.tofile() under unclear circumstances writes out things in strange
# order that is neither C or F, so we have the use this explicit conversion to get the result
# which we expect
def write_np_array(the_array, the_file_path):
  c_bytes = the_array.astype(np.float32).tobytes("C")
  with open(the_file_path, "wb") as f:
    f.write(c_bytes)

# The original model has batch normalization layers. We will now create
# a new model without batch norm. We will fold the parameters for each
# batch norm layer into the conv layer before it, so that we don't have
# to perform the batch normalization at inference time.
#
# Convert the weights and biases to Metal format.

def export_layers(the_model, model_name, dst_path):
  if not os.path.exists(dst_path):
    print('Creating output dir  "{}" for weights and biases'.format(dst_path))
    os.mkdir(dst_path)
   

  inbounds = find_inbound_layers(the_model)

  shape_info = {}
  for layer in the_model.layers:
    weights = layer.get_weights()
    if weights and len(weights):
      dprint("Exporting weights for layer "+layer.name+" type " + str(type(layer)))
      ddprint("layer config:" +pretty(layer.get_config())) 
      dprint("Layer '{}' has {} weight arrays".format(layer.name, len(weights)))
    for i, w in enumerate(weights):
      if i % 2 == 0:
        # In "th" format convolutional kernels have the shape (depth, input_depth, rows, cols)
        # In "tf" format convolutional kernels have the shape (rows, cols, input_depth, depth)
        #                              aka shape (height, width, inputChannels, outputChannels)
        ddprint("Keras weights shape:"+str(w.shape))
        outfile = "{}-{}.weights.bin".format(model_name,layer.name)
        outpath = os.path.join(dst_path, outfile)
        if type(layer) == Conv2D:
          ddprint("Converting weights to metal for Conv2D layer") 
          tw = w
          #if layer.get_config()["data_format"] == "channels_last":
          #elif layer.get_config()["data_format"] == "channels_first":
          #else:
          #  raise RuntimeError("unknown kernel weight format")
          # metal wants:  weight[ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels / groups ]
          tw = w.transpose(3, 0, 1, 2)
          ddprint("Metal conv weights shape:"+str(tw.shape))
          write_np_array(tw, outpath)
          shape_info[outfile] = tw.shape
        elif type(layer) == Dense:
          ddprint("Converting weights to metal for Dense layer") 
          # In metal there is no need for a flatten layer when CNN is fed to a dense layer,
          # but the weights must be properly arranges depending on the input layer:
          # The metal dense weights array: The number of entries is =
          # inputFeatureChannels * outputFeatureChannels * kernelHeight * kernelWidth
          # The layout of filter weight is so that it can be reinterpreted as 4D tensor (array)
          # weight[ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels / groups ]
          # where kernelHeight and kernelWidth are equal to the width and height of the input shape
          # In Metal, the output of the dense layer has width=1 height=1 and channnels=outputchannels
          input_channels =  w.shape[0]
          output_channels = w.shape[1]
          ddprint("input_channels=",input_channels)
          ddprint("output_channels=",output_channels)
          input_layer = the_model.get_layer(inbounds[layer.name][0])
          ddprint("input_layer.name",input_layer.name)
          metal_weights = None
          if type(input_layer) == Flatten:
            ddprint("Keras model has a Flatten layer, lets get the shape of its input")
            input_layer = the_model.get_layer(inbounds[input_layer.name][0])
            ddprint("effective input_layer.name",input_layer.name)
          fc_in_shape = input_layer.output_shape
          ddprint("fc_in_shape",str(fc_in_shape))
          if type(input_layer) != Dense and layer.__class__.__name__ == "Concatenate":
            ddprint("Keras model input to dense layer is not a Dense or Concatenate layer, so we need to arrange the weigts to its output shape")
            conv_outputs = fc_in_shape[3]
            conv_height = fc_in_shape[2]
            output_width = fc_in_shape[1]
            fc_shape = (output_channels, conv_outputs, conv_height, output_width)
            ddprint("fc_shape",str(fc_shape))
            rw = w.transpose().reshape(fc_shape)
            ddprint("intermediate weights shape rw:"+str(rw.shape))
            rwt = rw.transpose(0, 2, 3, 1)
            ddprint("intermediate weights shape rwt:"+str(rwt.shape))
            rrwt = rw.reshape(output_channels, input_channels)
            metal_weights = rrwt
            ''' 
            channels_in = 7*7*50
            channels_out = 320
            fc_shape = (320, 50, 7, 7)
            metal_weights = trained_weights.reshape(fc_shape).transpose(0, 2, 3, 1).reshape(channels_out, channels_in)
            '''
          else:
            ddprint("Keras model input to Dense layer is a another Dense layer")
            metal_weights = w.transpose()
          ddprint("Metal weights shape:"+str(metal_weights.shape))
          write_np_array(metal_weights, outpath)
          shape_info[outfile] = metal_weights.shape
        else:
          # Keras Dense weight have shape (inputs, outputs)
          # Metal Dense weight have shape (outputs, inputs)
          tw = w.transpose()
          ddprint("Saving weights for other layer than Conv2D or Dense, keras shape:"+
                 str(w.shape)+", metal shape:"+str(tw.shape))
          write_np_array(tw, outpath)
          shape_info[outfile] = tw.shape
      else:
        ddprint("Biases shape:"+str(w.shape))
        outfile = "{}-{}.biases.bin".format(model_name,layer.name)
        outpath = os.path.join(dst_path, outfile)
        write_np_array(w, outpath)
        #w.tofile(outpath)
        shape_info[outfile] = tw.shape
  with open(os.path.join(dst_path, "shapes.json"), "w") as json_file:
    print(pretty(shape_info), file=json_file)

if not QUICK_RUN:
  print("Exporting weights for new model to '{}'".format(PARAMETERS_OUT_DIR))
  export_layers(new_model, model_name, PARAMETERS_OUT_DIR)

if EXPORT_ORIGINAL_WEIGHTS:
  print("Exporting weights for original model to '{}'".format(PARAMETERS_ORIG_OUT_DIR))
  export_layers(model, model_name, PARAMETERS_ORIG_OUT_DIR)

class_of_layer = {}
layer_with_name = {}

print("Creating swift code for new model")

for layer in new_model.layers:
  class_of_layer[layer.name] = layer.__class__.__name__
  layer_with_name[layer.name] = layer

# returns a set of activations
def gather_activations(model):
  activations = set() 
  params_of_activation = {}
  activation_of_layer = {} 
  for index, layer in enumerate(model.layers):
    params = []
    if "activation" in layer.get_config():
      activation = layer.get_config()['activation']
      activations.add(activation)
      activation_of_layer[layer_name(layer)] = activation
      if activation == "linear" or activation == "softmax":
        params = {"metal_func": "nil",
                  "swift_prefix": "nil"}
      elif activation == "relu":
        params = {"metal_func": "MPSCNNNeuronReLU(device: device, a: 0)",
                  "swift_prefix": "relu"}
      params_of_activation[activation] = params
    elif type(layer) in [LeakyReLU, ELU]:
      param = float(layer.get_config()['alpha'])
      activation = "{}(alpha={:.5f})".format(layer.__class__.__name__,param)
      activations.add(activation)
      activation_of_layer[layer_name(layer)] = activation
      if type(layer) == LeakyReLU:
        params = {"metal_func": "MPSCNNNeuronReLU(device: device, a: {:.5f})".format(param),
                  "swift_prefix": "leaky"}
      elif type(layer) == ELU:
        params = {"metal_func": "MPSCNNNeuronELU(device: device, a: {:.5f})".format(param),
                  "swift_prefix": "elu"}
      params_of_activation[activation] = params
  return activations, params_of_activation, activation_of_layer

activations, params_of_activation, activation_of_layer = gather_activations(new_model)
dprint("activations:"+str(activations))
ddprint("params_of_activation:"+pretty(params_of_activation))
ddprint("activation_of_layer:"+pretty(activation_of_layer))

# transforms a dict of lists of inbounds layer names into a
# dict of lists of outbound destination names
def outbound_layers(inbound_layers_by_name):
  outbound = {}
  for name, inbounds in inbound_layers_by_name.items():
    for inbound in inbounds:
      if not inbound in outbound:
        outbound[inbound] = []
      outbound[inbound].append(name)
  for name, inbounds in inbound_layers_by_name.items():
    if not name in outbound:
      outbound[name] = []
  return outbound

# returns a dictionary of the activation layers in the new keras model
# and the correspondig Conv layers with activations that shall
# be referenced in the metal model instead because in manny case there
# is no need for separate activation in the metal model
def replaced_activation_layers(inbound_by_name, model):
  replacements = {}
  for index, layer in enumerate(model.layers):
    if type(layer) in [LeakyReLU, ELU, Activation] and\
      'swift_prefix' in params_of_activation[activation_of_layer[layer.name]]:
      inbound = inbound_by_name[layer.name][0]
      inbound_layer = model.get_layer(inbound)
      if type(inbound_layer) == Conv2D:
        replacements[layer.name] = inbound_layer.name
    #if type(layer) in [Flatten]:
    #  replacements[layer.name] = inbound_layer.name

  return replacements


# returns a map to list of connections (can be names of inputs or outputs) that 
# take the places of the gone leaky relus in the new topology for metal
def collapsed_layers(connections, replacements):
  result = {}
  for layer, edges in connections.items():
    if not layer in replacements:
      result[layer] = []
      for edge in edges:
        if edge in replacements:
          result[layer].append(replacements[edge])
        else:
          result[layer].append(edge)

  return result

# set up various maps to help with our new metal network topology
orig_inbound_by_name = find_inbound_layers(new_model)
ddprint("orig_inbound_by_name:"+str(orig_inbound_by_name))
ddprint()

replaced_layers = replaced_activation_layers(orig_inbound_by_name, new_model)
ddprint("replaced_layers:"+str(replaced_layers))
ddprint()

# key-value reversed version of replaced_layers
original_activation_layer_of = {v:k for k,v in replaced_layers.items()}
ddprint("original_activation_layer_of:"+str(original_activation_layer_of))

inbound_by_name = collapsed_layers(orig_inbound_by_name, replaced_layers)
ddprint("inbound_by_name:")
ddprint(pretty(inbound_by_name))
ddprint()

outbound_by_name = outbound_layers(inbound_by_name)
ddprint("outbound_by_name:")
ddprint(pretty(outbound_by_name))
ddprint()

class ChainSection(dict):
  def __init__(self, name, layers):
    self.name = name
    self.layers = layers
    dict.__init__(self, name=name, layers=layers)
    dddprint("New ChainSection:"+pretty(self))

class ConcatSection(dict):
  def __init__(self, name, inputs):
    self.name = name
    self.layers = inputs
    dict.__init__(self, name=name, inputs=inputs)#
    dddprint("New ConcatSection:"+pretty(self))

# This function traverses the net recursively building chains
# of layers that start and end at points where the graph splits
# or joins because we need to reference layers with multiple
# outputs when we functionally build the Forge graph
# We also need to gather inputs for concatenate layers separately
# because the DSL has a different signature when concatenating
#
# The returned result will be an array of sections
# Each section is either a ChainSection or ConcatSection with a name
# that corresponds to the output of the chain and
# ann array of layers that make up the chain or
# shall be concatenated
#
def chain_layers(start_name, inbound_by_name, outbound_by_name, sections, current_chain,visited=set()):
  dddprint("chain_layers: check chain starting at '"+start_name+"'")
  dddprint("chain_layers: visited '"+str(visited)+"'")
  if start_name in visited:
    dddprint("chain_layers: already visited chain starting at '"+start_name+"'")
    return
  visited.add(start_name)
  end = False
  current = start_name
  while not end:
    dddprint("chain_layers: current: '"+current+"'")
    if len(inbound_by_name[current]) > 1:
      # we have more that one input, so we add a concat section
      sections[current] = ConcatSection(current, inbound_by_name[current])
      end = True # a concat section also needs the output so the chain ends here
      # continue now chains
      for output in outbound_by_name[current]:
        chain_layers(output, inbound_by_name, outbound_by_name, sections, [current], visited)
    elif len(outbound_by_name[current]) > 1:
      # section is referenced by more than one layer, so the chain also ends
      current_chain.append(current)
      sections[current] = ChainSection(current, current_chain)
      end = True
      for output in outbound_by_name[current]:
        chain_layers(output, inbound_by_name, outbound_by_name, sections, [current], visited)
    elif len(outbound_by_name[current]) == 0:
      # we reached an output layer with no further layers
      current_chain.append(current)
      sections[current] = ChainSection(current, current_chain)
      end = True
    elif len(outbound_by_name[current]) == 1:
      # a normal node in a chain, append 
      current_chain.append(current)
      if len(inbound_by_name[outbound_by_name[current][0]]) > 1:
        # next node will be a concat node, so we create and finsish the chain
        sections[current] = ChainSection(current, current_chain)
        current_chain = []
      current = outbound_by_name[current][0]

# find layers without connections
def find_inout(inoutbound_by_name):
  results = []
  for key, inoutbounds in inoutbound_by_name.items():
    if len(inoutbounds) == 0:
      results.append(key)
  return results

input_layers = find_inout(inbound_by_name)
output_layers = find_inout(outbound_by_name)

dprint("Input layers:"+str(input_layers))
dprint("Output layers:"+str(output_layers))

# compute all our sections
section_dict = {}
chain_layers(input_layers[0], inbound_by_name, outbound_by_name, section_dict,[])
dddprint(pretty(section_dict))

# walk through all section and replace inputs of concat layers that have another
# concat layer as input with the inputs of this layer
def consolidate_concats(section_dict, inbound_by_name, outbound_by_name):
  raw_sections = []
  replaced_concat_of_concat = {}
  for section in section_dict.values():
    ddprint("Consolidating section '{}'".format(section.name))
    if isinstance(section, ConcatSection) and class_of_layer[section.name] == "Concatenate":
      out_bounds = outbound_by_name[section.name]
      goes_into_other_concat = any(class_of_layer[layer] == "Concatenate" for layer in out_bounds)
      has_concat_inputs = any(class_of_layer[layer] == "Concatenate" for layer in section.layers)
      ddprint("Consolidating section "+section.name,"goes_into_other_concat",goes_into_other_concat,"has_concat_inputs", has_concat_inputs)
      if goes_into_other_concat and has_concat_inputs:
        raise RuntimeError("Can't yet deal with multi-layered Concats")
      if not goes_into_other_concat:
        if not has_concat_inputs:
          ddprint("Consolidating section "+section.name," has no concat inputs, just copying")
          raw_sections.append(section)
        else:
          # insert all the new inputs
          new_section_inputs = []
          for input_name in section.layers:
            ddprint("Consolidating section "+section.name,", checking input",input_name)
            input_layer = layer_with_name[input_name]
            input_class = class_of_layer[input_layer.name]
            if input_class == "Concatenate":
              other_inputs = section_dict[input_name].layers
              ddprint("Consolidating section "+section.name,"extending with ",other_inputs)
              new_section_inputs.extend(other_inputs)
              replaced_concat_of_concat[input_name] = section.name
            else:
              ddprint("Consolidating section "+section.name,"appending ",input_name)
              new_section_inputs.append(input_name)
          new_section = ConcatSection(section.name, new_section_inputs)
          raw_sections.append(new_section)
    else:
      raw_sections.append(section)
  # replace all the names of concats that no longer exist with the new names
  for section in raw_sections:
    ddprint("replacing names in section:", pretty(section))
    if section.name in replaced_concat_of_concat:
      ddprint("replacing name of section {} with {}".format(section.name, replaced_concat_of_concat[section.name]))
      section.name = replaced_concat_of_concat[section.name]
    for index, layer in enumerate(section.layers):
      if layer in replaced_concat_of_concat:
        section.layers[index] = replaced_concat_of_concat[layer]
  return raw_sections, replaced_concat_of_concat
        
raw_sections, replaced_concat_of_concat = consolidate_concats(section_dict, inbound_by_name, outbound_by_name)

ddprint("replaced_concat_of_concat:\n"+pretty(replaced_concat_of_concat))
dddprint("raw_sections:\n"+pretty(raw_sections))

# returns a list of all sections that will have to be defined
# as referenceable constants
def defined_sections(sections):
  result = []
  for section in sections:
    result.append(section.name)
  return result 

defined = defined_sections(raw_sections)
ddprint("defined = "+str(defined))

# return a dict containing a list of names that are referenced
# by a each section; the third item in the section "head"
# acts as name for a section
def referencing_sections(sections):
  result = {}
  for referencing_section in sections:
    referencing = []
    for reference in referencing_section.layers:
      if reference in defined and reference != referencing_section.name:
        referencing.append(reference)
    result[referencing_section.name] = referencing
    ddprint(referencing_section.name+" is referencing "+str(referencing))
    #print("result ="+str(result))
  return result

# The sections generated by our "chain_sections()" traversal are not
# necessarily in the order we need them to declare them,
# so we may have to sort them
def sort_sections(sections):
  referencing = referencing_sections(sections)

  order = []

  # find sections that do not reference others 
  # and therefore can be declared first
  for name in referencing:
    if len(referencing[name]) == 0:
      order.append(name)

  dprint("Sections not referencing any other section (should contain all inputs):"+str(order))

  # repeat iterating over all sections that reference others,
  # adding every section that has all its referecens already
  # satisfied until we can't add a new section
  changed = True
  while changed:
    ddprint("\nStart ordering pass:")
    changed = False
    for name in referencing:
      ddprint("Checking references of section:"+ name)
      if not name in order:
        # we have not yet resolved the references for this section 
        ok = True
        # check all the references in this section
        # if one is not resolved, we skip this section for later
        for ref in referencing[name]:
          #print("ref={}, referencing[{}]={}".format(ref, name, referencing[name]))
          if not ref in order:
            ddprint("reference "+ ref + " is not yet defined")
            ok = False
            break
        if ok:    
          ddprint("All references of section '"+ name+"' are defined")
          order.append(name)
          #print("order="+str(order))
          changed = True

  # We could not add any more sections, so we should have sorted them all
  # if not, we can't do anything better
  if len(order) != len(sections):
    # there seem to be some unresolved references, check which ones
    print("sections="+pretty(sections))
    print("order so far="+str(order))
    print("Total count:", len(sections))
    print("Resolved count:", len(order))
    assert False, "Could not resolve all references"

  # so far we only gather section names by order, now bring the sections
  # themselves in suitable order
  result = []
  for name in order:
    for section in sections:
      if section.name == name:
        result.append(section)

  return result

referencing = referencing_sections(raw_sections)
#print(json.dumps(referencing ,sort_keys=False, indent=4))

sections = sort_sections(raw_sections)
ddprint("Sections ordered properly by referencing:")
ddprint(pretty(sections))

# stub for keras -> metal activation function translation
def translated_activation(activation):
  translation = {"linear":"nil"}
  assert  activation in translation, "Unknown activation:"+activation
  return translation[activation]

#generate swift source for Forge network code

# return the name of the activation layer even if the layer
# may have been merged into another (e.g. Conv) layer
def activation_layer(name):
  if name in original_activation_layer_of:
    return original_activation_layer_of[name]
  else:
    return name

def to_swift(value):
  if isinstance(value,bool):
    return "true" if value else "false"
  return str(value)

def to_swift_enum(enum):
  translation = { "same": ".same",
                  "valid": ".valid"}
  if enum in translation:
    return translation[enum]
  else:
    raise RuntimeError("Can't convert keras enum:"+enum)

def swap_2_coords(array2):
  return (array2[1], array2[0])

def index_0(value):
  return value[0]

def index_1(value):
  return value[1]

def index_2(value):
  return value[2]

def index_3(value):
  return value[3]

# hack for resnet with undefined input shape
def index_1_or_299(value):
  if value[1] != None:
    return value[1]
  return 299

def index_2_or_299(value):
  if value[2] != None:
    return value[2]
  return 299


def quote_string(name):
  return '"{}"'.format(name)

def is_function(arg):
  return callable(arg)

# Base class for all the source generation classes
class ForgeLayer():
  def __init__(self, layer, forge_class, params):
    self.layer = layer
    self.cfg = layer.get_config()
    self.name = layer.name
    self.forge_class = forge_class
    self.params = params
  def swift_source(self):
    ddprint(str(self.params))
    line = "let {} = {}(".format(self.name, self.forge_class)
    i = 1
    for swift_param in self.params:
      origin = self.params[swift_param]
      ddprint("swift_param:'"+swift_param+"', origin:'"+str(origin)+"'")
      cfg_name = origin[0] # name of the keras config item
      # third argument is a precomputed value if first argument is ""
      # or a conversion function if present
      converter = to_swift if len(origin) < 3 or not is_function(origin[2]) else origin[2]
      val = converter(self.cfg[cfg_name]) if cfg_name != "" else origin[2]
      # do not emit code if actual value is default value
      default_val = origin[1] if len(origin) > 1 else None
      if not default_val or to_swift(default_val) != val:
        if not swift_param[0]=='_':
          arg = "{}: {}".format(swift_param, val)
        else:
          arg = val
        if i > 1:
          line += ", "
        line += arg 
        i += 1
    line += ")"
    ddprint(line)
    return [line]    

# Generate Forge layer "Convolution":
#
# init(kernel: (Int, Int),
#    channels: Int,
#    stride: (Int, Int) = (1, 1),
#    padding: PaddingType = .same,
#    activation: MPSCNNNeuron? = nil,
#    useBias: Bool = true,
#    name: String = "")
#
class ForgeConv2D(ForgeLayer):
  def __init__(self, layer, var_name_of_activation):
    params = OrderedDict([ 
      ("kernel"    , ("kernel_size","",swap_2_coords)),
      ("channels"  , ("filters",)),
      ("stride"    , ("strides", (1, 1))),
      ("padding"   , ("padding", ".same", to_swift_enum)),
      ("useBias"   , ("use_bias", True)),
      ("activation", ("", "nil", 
        var_name_of_activation[activation_of_layer[activation_layer(layer.name)]])),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Convolution", params)

# Generate Forge layer "Activation"
#init(_ activation: MPSCNNNeuron, name: String = "")
class ForgeActivation(ForgeLayer):
  def __init__(self, layer, var_name_of_activation):
    params = OrderedDict([ 
      ("_activation", ("", "", 
        var_name_of_activation[activation_of_layer[activation_layer(layer.name)]])),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Activation", params)


# Generate Forge class "MaxPooling":
#
# init( kernel: (Int, Int),
#  stride: (Int, Int),
#  padding: PaddingType = .valid,
#  edgeMode: MPSImageEdgeMode = .clamp,
#  name: String = "")
#
class ForgePooling(ForgeLayer):
  def __init__(self, layer, forge_class):
    params = OrderedDict([ 
      ("kernel"  , ("pool_size",)),
      ("stride"  , ("strides",)),
      ("padding" , ("padding", ".valid", to_swift_enum)),
      ("edgeMode", ("", ".clamp", ".clamp")),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, forge_class, params)

class ForgeMaxPooling2D(ForgePooling):
  def __init__(self, layer):
    super().__init__(layer, "MaxPooling")

class ForgeAveragePooling2D(ForgePooling):
  def __init__(self, layer):
    super().__init__(layer, "AveragePooling")

class ForgeGlobalAveragePooling2D(ForgeLayer):
  def __init__(self, layer):
    params = OrderedDict([ 
      ("name"    , ("name","",quote_string)),
      ("useBias"   , ("", True, "false"))
    ])
    super().__init__(layer, "GlobalAveragePooling", params)

def padding_list(padding):
  return str((padding[0][0],padding[0][1],padding[1][0],padding[1][1]))

class ForgeZeroPadding2D(ForgeLayer):
  def __init__(self, layer):
    params = OrderedDict([ 
      ("tblr_padding", ("padding", None, padding_list)),
      ("name"        , ("name","",quote_string))
    ])
    super().__init__(layer, "ZeroPadding", params)

class ForgeSpaceToDepthX2(ForgeLayer):
  def __init__(self, layer):
    params = OrderedDict([ 
      ("name"        , ("name","",quote_string))
    ])
    super().__init__(layer, "SpaceToDepthX2", params)

# Generate Forge class "Dense":
#
# init(neurons: Int,
#   activation: MPSCNNNeuron? = nil,
#   useBias: Bool = true,
#   name: String = "")
class ForgeDense(ForgeLayer):
  def __init__(self, layer, var_name_of_activation):
    params = OrderedDict([ 
      ("neurons" , ("units",)),
      ("useBias" , ("use_bias", True)),
      ("activation", ("", "nil", 
        var_name_of_activation[activation_of_layer[activation_layer(layer.name)]])),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Dense", params)

# Forge supports only one input so far
input_already_defined = False

class ForgeInput(ForgeLayer):
  def __init__(self, layer):
    params = OrderedDict([ 
      ("width" , ("batch_input_shape", None, index_1_or_299)),
      ("height" ,("batch_input_shape", None, index_2_or_299))
      #("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Input", params)
  
  def swift_source(self):
    global input_already_defined
    if  input_already_defined:
      raise RuntimeError("Forge supports only one input but network has multiple inputs")
    src = []
    src.append("let input = Input()")
    if SWAP_INPUT_IMAGE_CHANNELS:
      src.append('let swap_channels = Custom(TransposeChannelsKernel('
                 'device: device, featureChannels: 3, permute: [2,1,0]), name: "rgb2bgr")')

    if SUBTRACT_IMAGENET_MEAN:
      src.append('let subtract_mean = Custom(SubtractMeanColor('
        'device: device, red: 123.68, green: 116.779, blue: 103.939, scale: 255.0), '
        'name: "subtract_mean")')

    # "let {} = input --> Resize(width: {}, height: {})"
    line = super().swift_source()[0].replace("Input", "input --> Resize")
    if SWAP_INPUT_IMAGE_CHANNELS:
      line += " -->  swap_channels"
    if SUBTRACT_IMAGENET_MEAN:
      line += " -->  subtract_mean"
    
    # inception V3 wants input scaled between -1 and 1, but we can not
    # know that from the model; there is nothing about that in the input config
    if SCALE_INPUT_TO_MINUS_1_AND_1:
      # "let {} = input --> Resize(width: {}, height: {}) --> Activation(input_scale)"
      line += ' --> Activation(input_scale, name: "input_scale")'
    src.append(line)
    ddprint(src)
    input_already_defined = True
    return src
   
# start swift source generation

swift_src = []

def capitalize(name):
  return "".join(list(map(lambda w: w.title(), name.split("_"))))

src_class_name = capitalize(model_name)
src_file_name = src_class_name+".swift"

date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

swift_src.append(
'''
//
//  Begin of autogenerated swift source code
//
//  {}
//
//  created {} by keras2metal.py
//
//  Converter wittenn by Pavel Mayer, Tognos GmbH, http://tognos.com/
//  based on YADK and Forge yolo2metal.py
//

import Foundation
import Forge
import MetalPerformanceShaders

final class {} {{

var model: Model
var device: MTLDevice
let name = "{}"

public init(device: MTLDevice) {{
  self.device = device
'''.format(src_file_name, date_string, src_class_name, model_name))

# declare activation functions

used_prefix_counter = {}
var_name_of_activation = {}

for activation in activations:
  params = params_of_activation[activation]
  
  if 'swift_prefix' in params and params['swift_prefix']:
    prefix = params['swift_prefix']

    if prefix != "nil" and prefix in used_prefix_counter.keys():
      next_prefix = used_prefix_counter[prefix] + 1
      prefix = prefix + "_" + str(used_prefix_counter[prefix])
      used_prefix_counter[prefix] = next_prefix
    else:
      used_prefix_counter[prefix] = 2

    var_name_of_activation[activation] = prefix
    if prefix != "nil":
      swift_src.append("let {} = {}".format(prefix, params['metal_func']))

# inception V3 wants input scaled between -1 and 1, but we can not
# know that from the model; there is nothing about that in the input config
if SCALE_INPUT_TO_MINUS_1_AND_1:
  swift_src.append("let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)")

source_obj_of = {}

for index, layer in enumerate(new_model.layers):
  name = layer.name
 
  dprint(str(index)+":" +layer.__class__.__name__+" name:"+name)
  dddprint(str(index)+"type::"+str(type(layer)))
  
  if layer.name in replaced_layers.keys():
    ddprint("Layer '"+name+"' has been replaced by activation in layer '"+replaced_layers[name])
  else:
    swift_obj = None
    if type(layer) == Conv2D:
      swift_obj = ForgeConv2D(layer, var_name_of_activation)
    elif type(layer) == keras.engine.topology.InputLayer:
      swift_obj = ForgeInput(layer)
    elif type(layer) == MaxPooling2D:
      swift_obj = ForgeMaxPooling2D(layer)
    elif type(layer) == AveragePooling2D:
      swift_obj = ForgeAveragePooling2D(layer)
    elif type(layer) == ZeroPadding2D:
      swift_obj = ForgeZeroPadding2D(layer)
    elif type(layer) == GlobalAveragePooling2D:
      swift_obj = ForgeGlobalAveragePooling2D(layer)
    elif type(layer) == Dense:
      swift_obj = ForgeDense(layer, var_name_of_activation)
    elif type(layer) in [Add, Multiply, Average, Maximum, Flatten]:
      source_obj_of[layer.name] = ForgeLayer(layer, layer.__class__.__name__, {})
      ddprint(str(type(layer))+" layer '"+name+"' will not be predefined here")
    elif layer.__class__.__name__ == "Concatenate":
      source_obj_of[layer.name] = ForgeLayer(layer, "Concatenate", {})
      ddprint("Concatenate layer '"+name+"' will not be predefined here")
    elif type(layer) == Activation:
      swift_obj = ForgeActivation(layer, var_name_of_activation)
    elif type(layer) == keras.layers.core.Lambda and layer.name.startswith("block"):
      # hack for inception resnet add/scale lambda layer
      ddprint("Add/Scale layer '"+name+"' will not be predefined here")
    elif type(layer) == keras.layers.core.Lambda and layer.name == "space_to_depth_x2":
      swift_obj = ForgeSpaceToDepthX2(layer)
    else:
      raise ValueError("Unknown layer type:"+str(type(layer))+"\nConfig:\n"+
                       pretty(layer.get_config()))
    if swift_obj:
      source_obj_of[layer.name] = swift_obj
      swift_src.extend(swift_obj.swift_source())

swift_src.append("")

swift_src.append("do {")

ignore_layer_types = ["Flatten"]

dot = None
if PLOT_MODELS:
  dot = pydot.Dot()
  dot.set('rankdir', 'TB') # set to 'LR' for horizontal graph
  dot.set('concentrate', True)
  dot.set_node_defaults(shape='record')

def addDotNode(layer_name, layer_type, attributes):
  #print("addDotNode:", layer_name, layer_type, attributes)
  forge_type = layer_type
  shape = ""
  if layer_name in source_obj_of:
    forge_type = source_obj_of[layer_name].forge_class
    shape = source_obj_of[layer_name].layer.output_shape
    #print("shape:", shape)
    if len(shape) == 4:
      shape = [shape[i] for i in (1,2,3,0)]
    elif len(shape) == 3:
      shape = [shape[i] for i in (1,2,0)]
    elif len(shape) == 2:
      shape = [shape[i] for i in (1,0)]
    else:
      raise RuntimeError("strange output shape:"+str(shape))

    shape = "\\n"+str(shape).replace("None","1")
  label = '{} : {}{}'.format(layer_name, forge_type,shape)
  #print("label:", label)
  node = pydot.Node(layer_name, label=label)
  if attributes["bold_frame"] == True:
    node.set("penwidth", 3)
  dot.add_node(node)

edges = []

def addDotEdge(from_node, to_node, attributes):
  #print("addDotEdge:", from_node, to_node, attributes)
  edge = pydot.Edge(from_node, to_node)
  edges.append(edge)

def addAllEdges():
  for edge in edges:
    dot.add_edge(edge)

for section in sections:
  var_name = section.name
  if isinstance(section, ChainSection):
    line = "let "+var_name + " = "
    char_offset = len(line)
    prev_id = None
    for index, layer_id in enumerate(section.layers):
      if class_of_layer[layer_id] not in ignore_layer_types:
        line += layer_id
        if len(line)-char_offset > 70:
          line +="\n        "
          char_offset = len(line)
        if index < len(section.layers)-1:
          line += " --> "
      if PLOT_MODELS:
        addDotNode(layer_id, class_of_layer[layer_id], {"bold_frame": layer_id == var_name, "pos":1})
        if prev_id:
          addDotEdge(prev_id, layer_id, {"pos":3})
        prev_id = layer_id
    swift_src.append(line)
  elif isinstance(section, ConcatSection):
    layer = layer_with_name[section.name]
    layer_class = class_of_layer[layer.name]
    merge_function = "Concatenate" if layer_class == "Concatenate" else "Collect"

    #print("layer:", str(layer))
    #print("merge_function:", merge_function)
    #print("layer_class:", layer_class)
    #print("section", pretty(section))
    collector = None
    if PLOT_MODELS:
      collector = section.name
      if merge_function != "Concatenate":
        collector = "for_"+section.name
        addDotNode(collector, "Collect", {"bold_frame": False ,"pos":2})

    line = "let "+var_name + " = {}([".format(merge_function)
    for index, layer_id in enumerate(section.layers):
      if PLOT_MODELS:
        addDotEdge(layer_id, collector, {"bold_frame": layer_id == var_name,"pos":4})
      if layer_class == "Lambda" and layer.name.startswith("block") and index == 1:
        # inception-resnet lambda layer, output = input_0 + scale * input_1
        scale = layer.get_config()["arguments"]["scale"]
        line += '{} --> Activation('\
                'MPSCNNNeuronLinear(device: device, a: {}, b: 0), name: "{}")'\
                .format(layer_id, scale, "scale_input1_"+layer.name)
      else:
        line += layer_id
      if index < len(section.layers)-1:
        line += ", "
    if merge_function != "Concatenate":
      line += '], name: "for_{}")'.format(layer.name)
      
      if layer_class == "Lambda" and layer.name.startswith("block"):
        # inception-resnet lambda layer, output = input_0 + scale * input_1
        scale = layer.get_config()["arguments"]["scale"]
        line += ' --> Add(name: "{}")'.format(layer.name)
      else:
        # Forge merge layers have the same class name as keras layers
        line += ' --> {}(name: "{}")'.format(layer_class,layer.name)
      if PLOT_MODELS:
        addDotEdge(collector, layer.name, {"pos":5})
    else:
      line += '], name: "{}")'.format(layer.name)

    swift_src.append(line)
  else:
    raise RuntimeError("Unknown section type")

# insert a softmax when the last layer is a ense keras layer with softmax activation
last_layer = model.layers[-1]
#print(pretty(last_layer.get_config()))
if type(last_layer) == Dense and last_layer.get_config()["activation"] == "softmax":
  swift_src.append("let output = {} --> Softmax()".format(output_layers[0]))
  if PLOT_MODELS:
    addDotNode("softmax_1", "Softmax", {"bold_frame": False, "pos":6})
    addDotEdge(output_layers[0], "softmax_1", {"pos":7})
    addDotNode("output", "Tensor", {"bold_frame": True, "pos":8})
    addDotEdge("softmax_1", "output",{"pos":9})
else:
  swift_src.append("let output = {}".format(output_layers[0]))
  if PLOT_MODELS:
    addDotNode("output", "Tensor", {"bold_frame": True, "pos":8})
    addDotEdge(output_layers[0], "output", {"pos":7})

swift_src.append("model = Model(input: input, output: output)")
swift_src.append("}")
swift_src.append("} // init")

swift_src.append("public func compile(inflightBuffers: Int) -> Bool {")

swift_src.append("return model.compile(device: device, inflightBuffers: inflightBuffers) { ")
swift_src.append("  name, count, type in ParameterLoaderBundle(name: name,")
swift_src.append("  count: count,")
swift_src.append('  prefix: "{}-",'.format(model_name))
swift_src.append('  suffix: type == .weights ? ".weights" : ".biases",')
swift_src.append('  ext: "bin")')
swift_src.append("}")
swift_src.append("} // func")
swift_src.append("} // class")
swift_src.append("")
swift_src.append("// end of autogenerated forge net generation code")

with open(SRC_OUT_DIR+"/"+src_file_name, "w") as src_file:
  for line in swift_src:
    print(line, file=src_file)
    if VERBOSE:
      print(line)

new_model.save(file_name_plus(model_path, "_nobn"))

if PLOT_MODELS:
  addAllEdges()
  to_file=changed_extension(file_name_plus(model_path,'_forge'),'.png')
  dot.write(to_file, format="png")
  if MORE_VERBOSE:
    print("dot file:")
    print("-------------------------------------------------------------")
    print(dot.to_string())
    print("-------------------------------------------------------------")

if not QUICK_RUN:
  # Make a prediction using the original model and also using the model that
  # has batch normalization removed, and check that the differences between
  # the two predictions are small enough. They seem to be smaller than 1e-4,
  # which is good enough for us, since we'll be using 16-bit floats anyway.

  print("Comparing models...")

  #image_data = np.random.random((1, 608, 608, 3)).astype('float32')

  test_shape = None
  if batch_input_shape[1] != None and batch_input_shape[2] != None:
    test_shape = (1, batch_input_shape[1], batch_input_shape[2],batch_input_shape[3])
  else:
    test_shape = (1, 299, 299, 3)
  image_data = np.random.random(test_shape).astype('float32')

  features = model.predict(image_data)
  features_auto = new_model.predict(image_data)

  ddprint("features.shape:"+str(features.shape))
  ddprint("features_auto.shape:"+str(features_auto.shape))

def compare_features_fast(features1, features2):
  error = np.abs(features1 - features2)
  max_error = np.max(error)
  avrg_error = np.sum(error)/features1.size
  min_1 = np.min(features1)
  min_2 = np.min(features2)
  max_1 = np.max(features1)
  max_2 = np.max(features2)
  avrg_1 = np.sum(features1)/features1.size
  avrg_2 = np.sum(features2)/features2.size
  print("Features original : avrg1:",avrg_1,"min1:",min_1,"max1:", max_1)
  print("Features new model: avrg2:",avrg_2,"min2:",min_2,"max2:",max_2)
  print("Error max:",max_error,"avrg:",avrg_error)

def compare_features_yolo(features1, features2):
  max_error = 0
  for i in range(features1.shape[1]):
    for j in range(features1.shape[2]):
      for k in range(features1.shape[3]):
        diff = np.abs(features1[0, i, j, k] - features2[0, i, j, k])
        max_error = max(max_error, diff)
        if diff > 1e-4:
          print(i, j, k, ":", features1[0, i, j, k], features2[0, i, j, k], diff)
  print("Largest error:", max_error)

if MODEL != "YOLO" and MODEL!="TINY_YOLO":
  if not QUICK_RUN:
    compare_features_fast(features, features_auto)
  if RUN_CLASSIFIER:
    predict_in_dir([model, new_model], IMAGE_DIR)
else:
  if not QUICK_RUN:
    compare_features_yolo(features, features_auto)

print("Done!")

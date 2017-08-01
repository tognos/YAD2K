# Conversion script for tiny-yolo-voc to Metal.
#
# The pretrained YOLOv2 model was made with the Darknet framework. You first
# need to convert it to a Keras model using YAD2K, and then yolo2metal.py can 
# convert the Keras model to Metal.
# 
# Required packages: python, numpy, h5py, pillow, tensorflow, keras.
#
# Download the tiny-yolo-voc.weights and tiny-yolo-voc.cfg files:
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

import os
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import layers

import copy

def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    #print("Folding bn "+bn_layer.__class__.__name__+":"+str(bn_layer.get_config())+" into conv "+conv_layer.__class__.__name__+":"+str(conv_layer.get_config())+"\n")
    conv_weights = conv_layer.get_weights()[0]

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]
    
    epsilon = 1e-3
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    return new_weights, new_bias

#model_path = "model_data/tiny-yolo-voc.h5"
model_path = "model_data/yolo.h5"

# Load the model that was exported by YAD2K.
model = load_model(model_path)
model.summary()

all_layers = []
prev_layer = None
prev_prev_layer = None
new_weights = []

print("Creating optimized model\n")


model_config = model.get_config()
layers_config = model_config['layers']

inbound_by_name = {}

for layer_dict in layers_config:
  layer_name = layer_dict['name']
  inbound_names = []
  inbound_nodes = layer_dict['inbound_nodes']
  if len(inbound_nodes) > 0:
    #print(str(inbound_nodes))
    inbound_node_list = inbound_nodes[0]
    for node in inbound_node_list:
      inbound_names.append(node[0])
    print("Layer name:"+layer_name+": Inbound:"+str(inbound_names))
  else:
    print("Layer name:"+layer_name+": No inbound layers")
  inbound_by_name[layer_name] = inbound_names


#import json
#print(json.dumps(model.get_config(),sort_keys=True, indent=4))


def layer_name(layer):
  return layer.get_config()['name']

def layer_clone(layer):
  return layers.deserialize({'class_name': layer.__class__.__name__,
                                    'config': layer.get_config()})

def replaced_name(the_name, replacements):
  if the_name in replacements:
    return replacements[the_name]
  else:
    return the_name

def replaced_name_list(the_names, replacements):
  result = []
  for name in the_names:
    result.append(replaced_name(name, replacements))
  return result


def input_layers_outputs(inbound_names, layer_by_name):
  #print("input_layers_outputs: inbound_names: "+str(inbound_names)+", all="+str(layer_by_name))
  print("input_layers_outputs: inbound_names: "+str(inbound_names))
  if len(inbound_names) == 1:
    result = layer_by_name[inbound_names[0]]
    print("input_layers_outputs: returning single result: "+str(result))
    return result;
  else:
    result = []
    for name in inbound_names:
      result.append(layer_by_name[name])
    print("input_layers: returning list result: "+str(result))
    return result


def orig_input_layers(inbound_names, model):
  if len(inbound_names) == 1:
    result = model.get_layer(name=inbound_names[0])
    print("orig_input_layers: returning single result: "+str(result))
    return result 
  else:
    result = []
    for iname in inbound_names:
      result.append(model.get_layer(name=iname))
    print("orig_input_layers: returning single result: "+str(result))
    return result


layer_by_name = {}
weights_by_name = {}
output_by_name = {}
replaced_layer = {}

def register_new_layer(the_name, the_layer, the_output):
  all_layers.append(the_output)
  layer_by_name[the_name] = the_layer
  output_by_name[the_name] = the_output

for index, layer in enumerate(model.layers):
  print("\n"+str(index)+":"+layer.__class__.__name__+":"+str(layer.get_config()))
  print("Layer name:"+layer_name(layer))

  inbounds = replaced_name_list(inbound_by_name[layer_name(layer)], replaced_layer)
  print("Inbounds:"+str(inbounds)+", len "+str(len(inbounds)))

  if len(inbounds) == 0:
    # create an input layer
    batch_input_shape = layer.get_config()['batch_input_shape']
    input_shape = batch_input_shape[1:]
    print(str(input_shape), str(batch_input_shape))
    new_layer = Input(shape=input_shape)
    register_new_layer(layer_name(layer), new_layer, new_layer)
  else:
    orig_inputs = orig_input_layers(inbounds, model)
    print("orig_inputs:"+str(orig_inputs))
    if not layer.get_weights():
      print("Layer '"+layer.__class__.__name__+" has no weights")
      new_layer = layer_clone(layer)
      inputs = input_layers_outputs(inbounds, output_by_name)
      register_new_layer(layer_name(layer), new_layer, new_layer(inputs))
    else:
      weight_shape = np.shape(layer.get_weights())
      print("Layer '"+layer.__class__.__name__+" weight shape:"+str(weight_shape))
      layer_done = False
      print("orig_inputs:"+str(orig_inputs)+", class "+str(orig_inputs.__class__))
      print("BNTEST:"+layer.__class__.__name__ +" "+ orig_inputs.__class__.__name__)
      
      if layer.__class__.__name__ == "BatchNormalization" and orig_inputs.__class__.__name__ == "Conv2D":
        # batchnorm following a conv2D layer, set folded weights for previous conv layer
        print("Folding batch norm layer")
        prev_orig_layer = orig_input_layers(inbounds, model)
        new_config = prev_orig_layer.get_config()
        new_config['use_bias'] = True
        new_layer = layers.deserialize({'class_name': prev_orig_layer.__class__.__name__,
                                  'config': new_config})
        prev_inbounds = replaced_name_list(inbound_by_name[layer_name(prev_orig_layer)],replaced_layer)
        inputs = input_layers_outputs(prev_inbounds, output_by_name)
        register_new_layer(layer_name(new_layer), new_layer, new_layer(inputs))
        print("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
        weights_by_name[layer_name(new_layer)] = fold_batch_norm(prev_orig_layer, layer)
        replaced_layer[layer_name(layer)] = layer_name(prev_orig_layer)
        #new_weights.append(fold_batch_norm(prev_orig_layer, layer))
        #new_weights.append(prev_orig_layer.get_weights())
        layer_done = True
      else:
        if orig_inputs.__class__.__name__ == "Conv2D":
          # conv without following batchnorm, set normal weights for previous conv layer
          print("Conv2d layer without following batchnorm")
          prev_orig_layer = model.get_layer(name=orig_inputs)
          new_layer = layer_clone(prev_orig_layer)
          prev_inbounds = replaced_name_list(inbound_by_name[layer_name(prev_orig_layer)],replaced_layer)
          inputs = input_layers_outputs(prev_inbounds, output_by_name)
          register_new_layer(layer_name(new_layer), new_layer, new_layer(inputs))
          weights_by_name[layer_name(new_layer)]=prev_orig_layer.get_weights()
      
      if not layer_done:
        # process all layer types except conv2d if not the last layer
        # if layer.__class__.__name__ != "Conv2D" or index + 1 == len(model.layers):
        if True:
          new_layer = layer_clone(layer)
          inputs = input_layers_outputs(inbounds, output_by_name)
          register_new_layer(layer_name(layer), new_layer, new_layer(inputs))
          print("appending new layer:"+new_layer.__class__.__name__)
          print("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
          weights_by_name[layer_name(layer)] = layer.get_weights()
          #print("adding weights for new layer type " + all_layers[-1].__class__.__name__)

new_model = Model(inputs=all_layers[0], outputs=all_layers[-1])
new_model.summary()

print(str(replaced_layer))

for layer_name, weights in weights_by_name.items():
  print("Setting weights for layer "+layer_name+" type " + layer_by_name[layer_name].__class__.__name__)
  print("weights     :"+str(np.shape(weights)))
  print("orig_weights:"+str(np.shape(layer_by_name[layer_name].get_weights())))
  layer_by_name[layer_name].set_weights(weights)


# The original model has batch normalization layers. We will now create
# a new model without batch norm. We will fold the parameters for each
# batch norm layer into the conv layer before it, so that we don't have
# to perform the batch normalization at inference time.
#
# All conv layers (except the last) have 3x3 kernel, stride 1, and "same"
# padding. Note that these conv layers did not have a bias in the original 
# model, but here they do get a bias (from the batch normalization).
#
# The last conv layer has a 1x1 kernel and identity activation.
#
# All max pool layers (except the last) have 2x2 kernel, stride 2, "valid" 
# padding. The last max pool layer has stride 1 and "same" padding.
#
# We still need to add the LeakyReLU activation as a separate layer, but 
# in Metal we can combine the LeakyReLU with the conv layer.

DO_STATIC_CONVERSION=False

if DO_STATIC_CONVERSION:
  model_nobn = Sequential()
  model_nobn.add(Conv2D(16, (3, 3), padding="same", input_shape=(416, 416, 3)))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D())
  model_nobn.add(Conv2D(32, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D())
  model_nobn.add(Conv2D(64, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D())
  model_nobn.add(Conv2D(128, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D())
  model_nobn.add(Conv2D(256, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D())
  model_nobn.add(Conv2D(512, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(MaxPooling2D(strides=(1, 1), padding="same"))
  model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(Conv2D(1024, (3, 3), padding="same"))
  model_nobn.add(LeakyReLU(alpha=0.1))
  model_nobn.add(Conv2D(125, (1, 1), padding="same", activation='linear'))

  W_nobn = []
  print("W_nobn shape:"+str(np.shape(W_nobn)))
  W_nobn.extend(fold_batch_norm(model.layers[1], model.layers[2]))
  print("W_nobn shape:"+str(np.shape(W_nobn)))
  W_nobn.extend(fold_batch_norm(model.layers[5], model.layers[6]))
  print("W_nobn shape:"+str(np.shape(W_nobn)))
  W_nobn.extend(fold_batch_norm(model.layers[9], model.layers[10]))
  W_nobn.extend(fold_batch_norm(model.layers[13], model.layers[14]))
  W_nobn.extend(fold_batch_norm(model.layers[17], model.layers[18]))
  W_nobn.extend(fold_batch_norm(model.layers[21], model.layers[22]))
  W_nobn.extend(fold_batch_norm(model.layers[25], model.layers[26]))
  W_nobn.extend(fold_batch_norm(model.layers[28], model.layers[29]))
  W_nobn.extend(model.layers[31].get_weights())
  print("W_nobn shape:"+str(np.shape(W_nobn)))
  #print(str(W_nobn))
  model_nobn.set_weights(W_nobn)

  print("model_nobn")
  model_nobn.summary()

  for index, layer in enumerate(model_nobn.layers):
    print("no_bn:"+str(index)+":"+layer.__class__.__name__+":"+str(layer.get_config()))
    print("newnb:"+str(index)+":"+new_model.layers[index+1].__class__.__name__+":"+str(new_model.layers[index+1].get_config())+"\n")
    print(str(np.shape(layer.get_weights())))
    print(str(np.shape(new_model.layers[index+1].get_weights())))

# Make a prediction using the original model and also using the model that
# has batch normalization removed, and check that the differences between
# the two predictions are small enough. They seem to be smaller than 1e-4,
# which is good enough for us, since we'll be using 16-bit floats anyway.

print("Comparing models...")

#image_data = np.random.random((1, 416, 416, 3)).astype('float32')
image_data = np.random.random((1, 608, 608, 3)).astype('float32')
features = model.predict(image_data)
if DO_STATIC_CONVERSION:
  features_nobn = model_nobn.predict(image_data)
features_auto = new_model.predict(image_data)

def compare_features(features1, features2):
  max_error = 0
  for i in range(features1.shape[1]):
    for j in range(features1.shape[2]):
      for k in range(features1.shape[3]):
        diff = np.abs(features1[0, i, j, k] - features2[0, i, j, k])
        max_error = max(max_error, diff)
        if diff > 1e-4:
          print(i, j, k, ":", features1[0, i, j, k], features2[0, i, j, k], diff)
  print("Largest error:", max_error)

if DO_STATIC_CONVERSION:
  compare_features(features, features_nobn)
compare_features(features, features_auto)

# Convert the weights and biases to Metal format.

if DO_STATIC_CONVERSION:
  
  print("\nConverting parameters...")

  dst_path = "Parameters"
  W = new_model.get_weights()
  for i, w in enumerate(W):
    j = i // 2 + 1
    print(w.shape)
    if i % 2 == 0:
      w.transpose(3, 0, 1, 2).tofile(os.path.join(dst_path, "conv%d_W.bin" % j))
    else:
      w.tofile(os.path.join(dst_path, "conv%d_b.bin" % j))

print("Done!")

'''
tiny-yolo-voc

Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 416, 416, 3)       0         
conv2d_1 (Conv2D)            (None, 416, 416, 16)      432       
batch_normalization_1 (Batch (None, 416, 416, 16)      64        
leaky_re_lu_1 (LeakyReLU)    (None, 416, 416, 16)      0         
max_pooling2d_1 (MaxPooling2 (None, 208, 208, 16)      0         
conv2d_2 (Conv2D)            (None, 208, 208, 32)      4608      
batch_normalization_2 (Batch (None, 208, 208, 32)      128       
leaky_re_lu_2 (LeakyReLU)    (None, 208, 208, 32)      0         
max_pooling2d_2 (MaxPooling2 (None, 104, 104, 32)      0         
conv2d_3 (Conv2D)            (None, 104, 104, 64)      18432     
batch_normalization_3 (Batch (None, 104, 104, 64)      256       
leaky_re_lu_3 (LeakyReLU)    (None, 104, 104, 64)      0         
max_pooling2d_3 (MaxPooling2 (None, 52, 52, 64)        0         
conv2d_4 (Conv2D)            (None, 52, 52, 128)       73728     
batch_normalization_4 (Batch (None, 52, 52, 128)       512       
leaky_re_lu_4 (LeakyReLU)    (None, 52, 52, 128)       0         
max_pooling2d_4 (MaxPooling2 (None, 26, 26, 128)       0         
conv2d_5 (Conv2D)            (None, 26, 26, 256)       294912    
batch_normalization_5 (Batch (None, 26, 26, 256)       1024      
leaky_re_lu_5 (LeakyReLU)    (None, 26, 26, 256)       0         
max_pooling2d_5 (MaxPooling2 (None, 13, 13, 256)       0         
conv2d_6 (Conv2D)            (None, 13, 13, 512)       1179648   
batch_normalization_6 (Batch (None, 13, 13, 512)       2048      
leaky_re_lu_6 (LeakyReLU)    (None, 13, 13, 512)       0         
max_pooling2d_6 (MaxPooling2 (None, 13, 13, 512)       0         
conv2d_7 (Conv2D)            (None, 13, 13, 1024)      4718592   
batch_normalization_7 (Batch (None, 13, 13, 1024)      4096      
leaky_re_lu_7 (LeakyReLU)    (None, 13, 13, 1024)      0         
conv2d_8 (Conv2D)            (None, 13, 13, 1024)      9437184   
batch_normalization_8 (Batch (None, 13, 13, 1024)      4096      
leaky_re_lu_8 (LeakyReLU)    (None, 13, 13, 1024)      0         
conv2d_9 (Conv2D)            (None, 13, 13, 125)       128125    
=================================================================
Total params: 15,867,885
Trainable params: 15,861,773
Non-trainable params: 6,112

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 608, 608, 3)   0                                            
conv2d_1 (Conv2D)                (None, 608, 608, 32)  864         input_1[0][0]                    
batch_normalization_1 (BatchNorm (None, 608, 608, 32)  128         conv2d_1[0][0]                   
leaky_re_lu_1 (LeakyReLU)        (None, 608, 608, 32)  0           batch_normalization_1[0][0]      
max_pooling2d_1 (MaxPooling2D)   (None, 304, 304, 32)  0           leaky_re_lu_1[0][0]              
conv2d_2 (Conv2D)                (None, 304, 304, 64)  18432       max_pooling2d_1[0][0]            
batch_normalization_2 (BatchNorm (None, 304, 304, 64)  256         conv2d_2[0][0]                   
leaky_re_lu_2 (LeakyReLU)        (None, 304, 304, 64)  0           batch_normalization_2[0][0]      
max_pooling2d_2 (MaxPooling2D)   (None, 152, 152, 64)  0           leaky_re_lu_2[0][0]              
conv2d_3 (Conv2D)                (None, 152, 152, 128) 73728       max_pooling2d_2[0][0]            
batch_normalization_3 (BatchNorm (None, 152, 152, 128) 512         conv2d_3[0][0]                   
leaky_re_lu_3 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_3[0][0]      
conv2d_4 (Conv2D)                (None, 152, 152, 64)  8192        leaky_re_lu_3[0][0]              
batch_normalization_4 (BatchNorm (None, 152, 152, 64)  256         conv2d_4[0][0]                   
leaky_re_lu_4 (LeakyReLU)        (None, 152, 152, 64)  0           batch_normalization_4[0][0]      
conv2d_5 (Conv2D)                (None, 152, 152, 128) 73728       leaky_re_lu_4[0][0]              
batch_normalization_5 (BatchNorm (None, 152, 152, 128) 512         conv2d_5[0][0]                   
leaky_re_lu_5 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_5[0][0]      
max_pooling2d_3 (MaxPooling2D)   (None, 76, 76, 128)   0           leaky_re_lu_5[0][0]              
conv2d_6 (Conv2D)                (None, 76, 76, 256)   294912      max_pooling2d_3[0][0]            
batch_normalization_6 (BatchNorm (None, 76, 76, 256)   1024        conv2d_6[0][0]                   
leaky_re_lu_6 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_6[0][0]      
conv2d_7 (Conv2D)                (None, 76, 76, 128)   32768       leaky_re_lu_6[0][0]              
batch_normalization_7 (BatchNorm (None, 76, 76, 128)   512         conv2d_7[0][0]                   
leaky_re_lu_7 (LeakyReLU)        (None, 76, 76, 128)   0           batch_normalization_7[0][0]      
conv2d_8 (Conv2D)                (None, 76, 76, 256)   294912      leaky_re_lu_7[0][0]              
batch_normalization_8 (BatchNorm (None, 76, 76, 256)   1024        conv2d_8[0][0]                   
leaky_re_lu_8 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_8[0][0]      
max_pooling2d_4 (MaxPooling2D)   (None, 38, 38, 256)   0           leaky_re_lu_8[0][0]              
conv2d_9 (Conv2D)                (None, 38, 38, 512)   1179648     max_pooling2d_4[0][0]            
batch_normalization_9 (BatchNorm (None, 38, 38, 512)   2048        conv2d_9[0][0]                   
leaky_re_lu_9 (LeakyReLU)        (None, 38, 38, 512)   0           batch_normalization_9[0][0]      
conv2d_10 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_9[0][0]              
batch_normalization_10 (BatchNor (None, 38, 38, 256)   1024        conv2d_10[0][0]                  
leaky_re_lu_10 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_10[0][0]     
conv2d_11 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_10[0][0]             
batch_normalization_11 (BatchNor (None, 38, 38, 512)   2048        conv2d_11[0][0]                  
leaky_re_lu_11 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_11[0][0]     
conv2d_12 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_11[0][0]             
batch_normalization_12 (BatchNor (None, 38, 38, 256)   1024        conv2d_12[0][0]                  
leaky_re_lu_12 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_12[0][0]     
conv2d_13 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_12[0][0]             
batch_normalization_13 (BatchNor (None, 38, 38, 512)   2048        conv2d_13[0][0]                  
leaky_re_lu_13 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_13[0][0]     
max_pooling2d_5 (MaxPooling2D)   (None, 19, 19, 512)   0           leaky_re_lu_13[0][0]             
conv2d_14 (Conv2D)               (None, 19, 19, 1024)  4718592     max_pooling2d_5[0][0]            
batch_normalization_14 (BatchNor (None, 19, 19, 1024)  4096        conv2d_14[0][0]                  
leaky_re_lu_14 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_14[0][0]     
conv2d_15 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_14[0][0]             
batch_normalization_15 (BatchNor (None, 19, 19, 512)   2048        conv2d_15[0][0]                  
leaky_re_lu_15 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_15[0][0]     
conv2d_16 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_15[0][0]             
batch_normalization_16 (BatchNor (None, 19, 19, 1024)  4096        conv2d_16[0][0]                  
leaky_re_lu_16 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_16[0][0]     
conv2d_17 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_16[0][0]             
batch_normalization_17 (BatchNor (None, 19, 19, 512)   2048        conv2d_17[0][0]                  
leaky_re_lu_17 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_17[0][0]     
conv2d_18 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_17[0][0]             
batch_normalization_18 (BatchNor (None, 19, 19, 1024)  4096        conv2d_18[0][0]                  
leaky_re_lu_18 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_18[0][0]     
conv2d_19 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_18[0][0]             
batch_normalization_19 (BatchNor (None, 19, 19, 1024)  4096        conv2d_19[0][0]                  
conv2d_21 (Conv2D)               (None, 38, 38, 64)    32768       leaky_re_lu_13[0][0]             
leaky_re_lu_19 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_19[0][0]     
batch_normalization_21 (BatchNor (None, 38, 38, 64)    256         conv2d_21[0][0]                  
conv2d_20 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_19[0][0]             
leaky_re_lu_21 (LeakyReLU)       (None, 38, 38, 64)    0           batch_normalization_21[0][0]     
batch_normalization_20 (BatchNor (None, 19, 19, 1024)  4096        conv2d_20[0][0]                  
space_to_depth_x2 (Lambda)       (None, 19, 19, 256)   0           leaky_re_lu_21[0][0]             
leaky_re_lu_20 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_20[0][0]     
concatenate_1 (Concatenate)      (None, 19, 19, 1280)  0           space_to_depth_x2[0][0]          
                                                                   leaky_re_lu_20[0][0]             
conv2d_22 (Conv2D)               (None, 19, 19, 1024)  11796480    concatenate_1[0][0]              
batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096        conv2d_22[0][0]                  
leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_22[0][0]     
conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625      leaky_re_lu_22[0][0]             
====================================================================================================
Total params: 50,983,561
Trainable params: 50,962,889
Non-trainable params: 20,672
____________________________________________________________________________________________________
'''



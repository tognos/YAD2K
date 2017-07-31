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

model_path = "model_data/tiny-yolo-voc.h5"

# Load the model that was exported by YAD2K.
model = load_model(model_path)
model.summary()

all_layers = []
prev_layer = None
prev_prev_layer = None
new_weights = []

print("Creating optimized model\n")

for index, layer in enumerate(model.layers):
  print(str(index)+":"+layer.__class__.__name__+":"+str(layer.get_config()))
  #print(str(index)+" weights:"+str(layer.get_weights())+"\n")
  new_layer = layers.deserialize({'class_name': layer.__class__.__name__,
                                  'config': layer.get_config()})
  if prev_layer is not None:
    print(str(index)+"prev_layer:"+prev_layer.__class__.__name__)
    if not layer.get_weights():
      print("Layer '"+layer.__class__.__name__+" has no weights")
      all_layers.append(new_layer(prev_layer))
      prev_layer = all_layers[-1]
      print(str(index)+"prev_layer(1) set to :"+prev_layer.__class__.__name__)
    else:
      weight_shape = np.shape(layer.get_weights())
      print("Layer '"+layer.__class__.__name__+" weight shape:"+str(weight_shape))
      layer_done = False
      if layer.__class__.__name__ == "BatchNormalization" and prev_orig_layer.__class__.__name__ == "Conv2D":
        # batchnorm following a conv2D layer, set folded weights for previous conv layer
        print("Folding batch norm layer")
        new_config = prev_orig_layer.get_config()
        new_config['use_bias'] = True
        new_layer = layers.deserialize({'class_name': prev_orig_layer.__class__.__name__,
                                  'config': new_config})
        all_layers.append((new_layer)(prev_layer))
        prev_layer = all_layers[-1]
        print("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
        new_weights.append(fold_batch_norm(prev_orig_layer, layer))
        #new_weights.append(prev_orig_layer.get_weights())
        layer_done = True
      else:
        if prev_orig_layer.__class__.__name__ == "Conv2D":
          # conv without following batchnorm, set normal weights for previous conv layer
          print("Conv2d layer without following batchnorm")
          print("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
          new_layer = layers.deserialize({'class_name': prev_orig_layer.__class__.__name__,
                                  'config': prev_orig_layer.get_config()})
          all_layers.append((new_layer)(prev_layer))
          new_weights.append(prev_orig_layer.get_weights())
      if not layer_done:
        # process all layer types except
        if layer.__class__.__name__ != "Conv2D" or index + 1 == len(model.layers):
          all_layers.append((new_layer)(prev_layer))
          print("appending new layer:"+new_layer.__class__.__name__)
          print("adding weights for new layer index "+str(len(all_layers))+" type " + new_layer.__class__.__name__)
          new_weights.append(layer.get_weights())
          #print("adding weights for new layer type " + all_layers[-1].__class__.__name__)
      prev_layer = all_layers[-1]
      print(str(index)+"prev_layer(2) set to :"+prev_layer.__class__.__name__)
  else:
    batch_input_shape = layer.get_config()['batch_input_shape']
    input_shape = batch_input_shape[1:]
    print(str(input_shape), str(batch_input_shape))
    new_layer = Input(shape=input_shape)
    all_layers.append(new_layer)
    #all_layers.append(Input(shape=(image_height, image_width, 3)))
    #all_layers.append(new_layer)
    prev_layer = all_layers[-1]
    print(str(index)+"prev_layer(3) set to :"+prev_layer.__class__.__name__)

  prev_new_layer = new_layer
  #print("clone:"+str(index)+":"+new_layer.__class__.__name__+":"+str(new_layer.get_config())+"\n")
  #print("clone:"+str(index)+":"+new_layer.__class__.__name__+"\n")
  prev_orig_layer = layer
  #print("new_weights shape:"+str(np.shape(new_weights))+"\n")

#print(str(new_weights))

new_model = Model(inputs=all_layers[0], outputs=all_layers[-1])
new_model.summary()

weight_index = 0
for index, layer in enumerate(new_model.layers):
  if layer.get_weights():
    print("Setting weights for layer index "+str(index)+", weight_index "+str(weight_index)+" type " + layer.__class__.__name__)
    layer.set_weights(new_weights[weight_index])
    weight_index += 1


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

image_data = np.random.random((1, 416, 416, 3)).astype('float32')
features = model.predict(image_data)
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

compare_features(features, features_nobn)
compare_features(features, features_auto)

# Convert the weights and biases to Metal format.

print("\nConverting parameters...")

dst_path = "Parameters"
W = model_nobn.get_weights()
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



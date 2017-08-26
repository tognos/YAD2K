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

DEBUG_OUT = True
MORE_DEBUG_OUT = False

def dprint(*args, **kwargs):
  if DEBUG_OUT:
    print(*args, **kwargs)

def ddprint(*args, **kwargs):
  if MORE_DEBUG_OUT:
    print(*args, **kwargs)


def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for 
       the previous layer."""
    ddprint("Folding bn "+bn_layer.__class__.__name__+":"+str(bn_layer.get_config())+" into conv "+conv_layer.__class__.__name__+":"+str(conv_layer.get_config())+"\n")
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

print("Creating optimized model (with batchnorm folding):\n")

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
      dprint("Layer name:"+layer_name+": Inbound:"+str(inbound_names))
    else:
      dprint("Layer name:"+layer_name+": No inbound layers")
    inbound_layer_by_name[layer_name] = inbound_names
  return inbound_layer_by_name


import json
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
    #print("orig_inputs:"+str(orig_inputs))
    if not layer.get_weights():
      # layer has no weights, so we just clone it and instantiate
      # and connect it using the keras functional API
      dprint("Layer '"+layer.__class__.__name__+" has no weights")
      new_layer = layer_clone(layer)
      inputs = input_layers_outputs(inbounds, output_by_name)
      register_new_layer(layer_name(layer), new_layer, new_layer(inputs))
    else:
      # layer has weights, so we might have to do some optimization
      weight_shape = np.shape(layer.get_weights())
      dprint("Layer '"+layer.__class__.__name__+" weight shape:"+str(weight_shape))
      layer_done = False
      #print("orig_inputs:"+str(orig_inputs)+", class "+str(orig_inputs.__class__))
      #print("BNTEST:"+layer.__class__.__name__ +" "+ orig_inputs.__class__.__name__)
      
      if layer.__class__.__name__ == "BatchNormalization" and orig_inputs.__class__.__name__ == "Conv2D":
        # batchnorm following a conv2D layer
        # we need to set folded weights for the previous conv layer
        # which also has not been created yet
        dprint("Folding batch norm layer")
        prev_orig_layer = orig_input_layers(inbounds, model)
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
      else:
        if orig_inputs.__class__.__name__ == "Conv2D":
          # conv without following batchnorm, set normal weights for previous conv layer
          dprint("Conv2d layer without following batchnorm")
          prev_orig_layer = model.get_layer(name=orig_inputs)
          new_layer = layer_clone(prev_orig_layer)
          prev_inbounds = replaced_name_list(inbound_by_name[layer_name(prev_orig_layer)],replaced_layer)
          inputs = input_layers_outputs(prev_inbounds, output_by_name)
          register_new_layer(layer_name(new_layer), new_layer, new_layer(inputs))
          weights_by_name[layer_name(new_layer)]=prev_orig_layer.get_weights()
      
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
new_model.summary()

ddprint("Replaced layers:"+str(replaced_layer))

# now actually set the weights for the new model layers
# we have to do it after model instantiation because
# the keras could not calculate the actual shapes
# before the model was completely set up
for layer_name_, weights in weights_by_name.items():
  print("Setting weights for layer "+layer_name_+" type " + layer_by_name[layer_name_].__class__.__name__)
  #print("weights     :"+str(np.shape(weights)))
  #print("orig_weights:"+str(np.shape(layer_by_name[layer_name_].get_weights())))
  layer_by_name[layer_name_].set_weights(weights)


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

# Convert the weights and biases to Metal format.

EXPORT_LAYERS = False
if EXPORT_LAYERS:
  for layer_name_, weights in weights_by_name.items():
    print("Exporting weights for layer "+layer_name_+" type " + layer_by_name[layer_name_].__class__.__name__)
    #print("weights     :"+str(np.shape(weights)))
    #print("orig_weights:"+str(np.shape(layer_by_name[layer_name_].get_weights())))

    dst_path = "Parameters"
    W = new_model.get_weights()
    for i, w in enumerate(weights):
      if i % 2 == 0:
        print("Weights shape:"+w.shape)
        outpath = os.path.join(dst_path, "%s.weights.bin" % layer_name_)
        w.transpose(3, 0, 1, 2).tofile(outpath)
      else:
        print("Biases shape:"+w.shape)
        outpath = os.path.join(dst_path, "%s.biases.bin" % layer_name_)
        w.tofile(outpath)

class_of_layer = {}

for layer in new_model.layers:
  class_of_layer[layer_name(layer)] = layer.__class__.__name__

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

# returns a dictionary of the LeakyReLUs layers in the new keras model
# and the correspondig Conv layers with leaky activations that shall
# be referenced in the metal model instead because there is no need
# for LeakyReLUs in the metal model
def replaced_leakyReLUs(inbound_by_name, model):
  replacements = {}
  for index, layer in enumerate(model.layers):
    if layer.__class__.__name__ == "LeakyReLU":
      inbound = inbound_by_name[layer_name(layer)][0]
      inbound_layer = model.get_layer(inbound)
      if inbound_layer.__class__.__name__ == "Conv2D":
        replacements[layer_name(layer)] = layer_name(inbound_layer)
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
dprint("orig_inbound_by_name:"+str(orig_inbound_by_name))
dprint()
replaced_layers = replaced_leakyReLUs(orig_inbound_by_name, new_model)
dprint("replaced_layers:"+str(replaced_layers))
dprint()
inbound_by_name = collapsed_layers(orig_inbound_by_name, replaced_layers)
dprint("inbound_by_name:")
dprint(json.dumps(inbound_by_name ,sort_keys=True, indent=4))
dprint()
outbound_by_name = outbound_layers(inbound_by_name)
dprint("outbound_by_name:")
dprint(json.dumps(outbound_by_name ,sort_keys=True, indent=4))
dprint()


# filters 
#
def filter_duplicates(existing, new):
  result = []
  for section in new:
    exists = False
    for existing_section in existing:
      if existing_section == section:
        exists = True
    if not exists:
      result.append(section)
  return result

#
# Note: The following code that creates the swift source is not well tested,
# and might fail for special topologies, and there may be better ways to do that,
# but it does the job for yolo
#

# This function traverses the net recursively building chains
# of layers that start and end at points where the graph splits
# or joins because we need to reference layers with multiple
# outputs when we functionally build the Forge graph
# We also need to gather inputs for concatenate layers separately
# because the DSL has different signature when concatenating
#
# The returned result will be an array of sections
# Each section is a list with four element:
# - A string tag that is either "CHAIN" or "CONCAT"
# - a start name
# - an end name
# - an array of the layer names that form the chain or input
#
# Each chain is uniquely identfied by the first three elements
# (tag, start name, end name)
#
# The recursive traversal can create duplicates which are simply
# filtered out during the traversal
#
def chain_layers(start_name, inbound_by_name, outbound_by_name):
  current = start_name
  result = []
  links = []
  end = False
  # iterate along connections where the layer has only one output and
  # and no more than one input
  while not end and len(outbound_by_name[current]) == 1 and\
        (len(inbound_by_name[current]) <= 1 or
        current == start_name):
    if current == start_name and len(inbound_by_name[current]) == 1:
      # When we start the chain, reference the input layer 
      # to the chain if there is one 
      links.append(inbound_by_name[current][0])
   
    links.append(current)
    
    if len(outbound_by_name[current]) > 0:
      # advance when there is another output
      current = outbound_by_name[current][0]
    else:
      # stop iterating when there is no output connection
      end = True

  # We are done with gathering the elements (links) for the sectionn
  if class_of_layer[current] != "Concatenate": 
    # We stopped at a concat layer, but is has not been yet added
    # because Python does not have do..while loops
    links.append(current)

  # add a "CHAIN" section
  section = ("CHAIN", start_name, links[-1], links)
  result.extend(filter_duplicates(result, [section]))
  if len(outbound_by_name[current]) > 1:
    # we have more than one one outbound connection, follow them all recursively
    for out in outbound_by_name[current]:
      result.extend(filter_duplicates(result, chain_layers(out, inbound_by_name, outbound_by_name)))

  if len(inbound_by_name[current]) > 1:
    # we have more that one input, so we create a "CONCAT" section
    section = ("CONCAT", current, current, inbound_by_name[current])
    result.append(section)
    result.extend(filter_duplicates(result,chain_layers(current, inbound_by_name, outbound_by_name)))

  return result

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
raw_sections = chain_layers(input_layers[0], inbound_by_name, outbound_by_name)
#print(json.dumps(raw_sections ,sort_keys=False, indent=4))

# returns a list of all sections that will have to be defined
# as referenceable constants
def defined_sections(sections):
  result = []
  for section in sections:
    result.append(section[2])
  return result 

defined = defined_sections(raw_sections)
dprint("defined = "+str(defined))

# return a dict containing a list of names that are referenced
# by a each section; the third item in the section "head"
# acts as name for a section
def referencing_sections(sections):
  result = {}
  for referencing_section in sections:
    referencing = []
    for reference in referencing_section[3]:
      if reference in defined and reference != referencing_section[2]:
        referencing.append(reference)
    result[referencing_section[2]] = referencing
    dprint(referencing_section[2]+" is referencing "+str(referencing))
    #print("result ="+str(result))
  return result

# The sections generated by our "chain_sections()" traversal are not
# necessarily in the order we need them to declare them,
# so we may have to sort them
def sort_sections(sections):
  referencing = referencing_sections(raw_sections)

  order = []

  # find sections that do not reference others 
  # and therefore can be declared first
  for name in referencing:
    if len(referencing[name]) == 0:
      order.append(name)

  dprint("Sections not referencing any other section:="+str(order))

  # repeat iterating over all sections that reference others,
  # adding every section that has all its referecens already
  # satisfied until we can't add a new section
  changed = True
  while changed:
    changed = False
    for name in referencing:
      #print("Checking "+ name)
      if not name in order:
        ok = True
        for ref in referencing[name]:
          #print("ref={}, referencing[{}]={}".format(ref, name, referencing[name]))
          if not ref in order:
            ok = False
            break
        if ok:    
          order.append(name)
          #print("order="+str(order))
          changed = True

  # We could not add any more sections, so we should have sorted them all
  # if not, we can't do anything better
  assert len(order) == len(sections), "Could not resolve all references"

  # so far we only gather section names by order, now bring the sections
  # in order
  result = []
  for name in order:
    for section in sections:
      if section[2] == name:
        result.append(section)

  return result

referencing = referencing_sections(raw_sections)
#print(json.dumps(referencing ,sort_keys=False, indent=4))

sections = sort_sections(raw_sections)
#sections = raw_sections
dprint("SORTED::::::::::::::::::::")
dprint(json.dumps(sections ,sort_keys=False, indent=4))

# stub for keras -> metal activation function translation
def translated_activation(activation):
  translation = {"linear":"nil"}
  assert  activation in translation, "Unknown activation:"+activation
  return translation[activation]

#generate swift source for Forge network code
'''
public init(kernel: (Int, Int),
    channels: Int,
    stride: (Int, Int) = (1, 1),
    padding: PaddingType = .same,
    activation: MPSCNNNeuron? = nil,
    useBias: Bool = true,
    name: String = "") {
'''
def conv2DSource(kernel, channels, stride, padding, activation, useBias, name):
#Convolution(kernel: (3, 3), channels: 16, activation: leaky, name: "conv1")
  line = 'let {} = Convolution(kernel: ({}, {}), channels: {}, stride: ({}, {}), padding: .{},'\
         ' activation: {}, name: "{}")'.format(name, kernel[0], kernel[1], channels, stride[0],
         stride[1], padding, activation, name)
  print(line)
  return line


'''
  kernel: (Int, Int),
  stride: (Int, Int),
  padding: PaddingType = .valid,
  edgeMode: MPSImageEdgeMode = .clamp,
  name: String = ""
'''
def maxPoolSource(kernel, stride, padding, edgeMode, name):
#Convolution(kernel: (3, 3), channels: 16, activation: leaky, name: "conv1")
  line = 'let {} = MaxPooling(kernel: ({}, {}), stride: ({}, {}), padding: .{},'\
         'edgeMode: {}, name: "{}")'.format(name, kernel[0], kernel[1], stride[0],
         stride[1], padding, edgeMode, name)
  print(line)
  return line

def conv2DsourceFromConfig(layer):
  cfg = layer.get_config()
  
  #print(json.dumps(config ,sort_keys=True, indent=4))
  activation = cfg["activation"]
  if layer_name(layer) in replaced_layers.values():
    activation = "leaky"
  else:
    activation = translated_activation(activation)
  return conv2DSource(cfg["kernel_size"], cfg["filters"], cfg["strides"], cfg["padding"],
                      activation, cfg["use_bias"], layer_name(layer))

def maxPoolsourceFromConfig(layer):
  cfg = layer.get_config()
  #print(json.dumps(config ,sort_keys=True, indent=4))
  return maxPoolSource(cfg["pool_size"], cfg["strides"], cfg["padding"],
                      ".clamp", layer_name(layer))


swift_src = []
swift_src.append("")
swift_src.append("// begin of autogenerated forge net generation code")
swift_src.append("")

class_of_layer = {}
swift_src.append("let leaky = MPSCNNNeuronReLU(device: device, a: 0.1)")
for input_name in input_layers:
  swift_src.append("let input = Input()")
  swift_src.append("let {} = input --> Resize(width: {}, height: {})"\
                  .format(input_name, batch_input_shape[1], batch_input_shape[2]))
  swift_src.append("")

for index, layer in enumerate(new_model.layers):
  print(str(index)+":"+layer.__class__.__name__+" name:"+layer_name(layer))
  class_of_layer[layer_name(layer)] = layer.__class__.__name__
  #print("\n"+str(index)+":"+layer.__class__.__name__+":"+str(layer.get_config()))
  #print("Layer name:"+layer_name(layer))
  inbound_layers = orig_input_layers(inbounds, new_model)
  if layer.__class__.__name__ == "Conv2D":
    swift_src.append(conv2DsourceFromConfig(layer))
  elif layer.__class__.__name__ == "MaxPooling2D":
    swift_src.append(maxPoolsourceFromConfig(layer))
  elif layer.__class__.__name__ == "Lambda" and layer.name == "space_to_depth_x2":
    swift_src.append('let {} = SpaceToDepthX2(name: "{}")'.format(layer.name, layer.name))

swift_src.append("")

swift_src.append("do {")

for section in sections:
  if (section[0] == "CHAIN"):
    var_name = section[2]
    line = "let "+var_name + " = "
    for index, layer_id in enumerate(section[3]):
      line += layer_id
      if index < len(section[3])-1:
        line += " --> "
    swift_src.append(line)
  elif section[0] == "CONCAT":
    var_name = section[1]
    line = "let "+var_name + " = Concatenate(["
    for index, layer_id in enumerate(section[3]):
      line += layer_id
      if index < len(section[3])-1:
        line += ", "
    line += "])"
    swift_src.append(line)
 
swift_src.append("let output = {}".format(output_layers[0]))
swift_src.append("model = Model(input: input, output: output)")
swift_src.append("}")
swift_src.append("let success = model.compile(device: device, inflightBuffers: inflightBuffers) { ")
swift_src.append("  name, count, type in ParameterLoaderBundle(name: name,")
swift_src.append("  count: count,")
swift_src.append('  suffix: type == .weights ? ".weights" : ".biases",')
swift_src.append('  ext: "bin")')
swift_src.append("}")
swift_src.append("")
swift_src.append("// end of autogenerated forge net generation code")

for line in swift_src:
  print(line)

new_model.save("model_data/yolo_nobn.h5")

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


# Make a prediction using the original model and also using the model that
# has batch normalization removed, and check that the differences between
# the two predictions are small enough. They seem to be smaller than 1e-4,
# which is good enough for us, since we'll be using 16-bit floats anyway.

print("Comparing models...")


#image_data = np.random.random((1, 416, 416, 3)).astype('float32')
#image_data = np.random.random((1, 608, 608, 3)).astype('float32')
test_shape = (1, batch_input_shape[1], batch_input_shape[2],batch_input_shape[3])
image_data = np.random.random(test_shape).astype('float32')

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



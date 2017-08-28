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
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, Input, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU, ThresholdedReLU
from keras.models import Model
from keras import layers

from keras.applications.inception_v3 import InceptionV3
from collections import OrderedDict
import copy

DEBUG_OUT = True
MORE_DEBUG_OUT = False

#MODEL="TINY_YOLO"
#MODEL="YOLO"
MODEL="INCEPTION_V3"

if MODEL == "YOLO":
  model_name = "yolo"
  model_path = "model_data/yolo.h5"
  model = load_model(model_path)

if MODEL == "TINY_YOLO":
  model_name = "tiny_yolo"
  model_path = "model_data/tiny-yolo-voc.h5"
  model = load_model(model_path)

if MODEL=="INCEPTION_V3":
  model_name = "inception_v3"
  model_path = "model_data/inception_v3.h5"
  model = InceptionV3(weights='imagenet')
  model.save(model_path)

def file_name_plus(path_name, name_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + name_extension + extension

def changed_extension(path_name, new_extension):
  path_file, extension = os.path.splitext(path_name)
  return path_file + new_extension

model.summary()
from keras.utils import plot_model
plot_model(model, to_file=changed_extension(model_path,'.png'))

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
plot_model(new_model, to_file=changed_extension(file_name_plus(model_path,'_new'),'.png'))

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
# Convert the weights and biases to Metal format.

def export_layers(model, model_name, dst_path="Parameters"):
  for layer in model.layers:
    weights = layer.get_weights()
    if weights and len(weights):
      dprint("Exporting weights for layer "+layer.name+" type " + str(type(layer)))
    for i, w in enumerate(weights):
      if i % 2 == 0:
        dprint("Weights shape:"+str(w.shape))
        outpath = os.path.join(dst_path, "{}-{}.weights.bin".format(model_name,layer.name))
        if type(layer) == Conv2D:
          w.transpose(3, 0, 1, 2).tofile(outpath)
        else:
          w.tofile(outpath)
      else:
        dprint("Biases shape:"+str(w.shape))
        outpath = os.path.join(dst_path, "{}-{}.biases.bin".format(model_name,layer.name))
        w.tofile(outpath)

export_layers(new_model, model_name)

'''
def export_layers(model, dst_path="Parameters"):
  W = new_model.get_weights()
  for layer_name_, weights in weights_by_name.items():
    print("Exporting weights for layer "+layer_name_+" type " + layer_by_name[layer_name_].__class__.__name__)
    #print("weights     :"+str(np.shape(weights)))
    #print("orig_weights:"+str(np.shape(layer_by_name[layer_name_].get_weights())))
    layer = layer_by_name[layer_name_]
      for i, w in enumerate(weights):
        if i % 2 == 0:
          dprint("Weights shape:"+str(w.shape))
          outpath = os.path.join(dst_path, "{}-{}.weights.bin".format(model_name,layer_name_))
          if type(layer) == Conv2D:
            w.transpose(3, 0, 1, 2).tofile(outpath)
          else:
            w.tofile(outpath)
        else:
          dprint("Biases shape:"+str(w.shape))
          outpath = os.path.join(dst_path, "{}-{}.biases.bin".format(model_name,layer_name_))
          w.tofile(outpath)
'''      
class_of_layer = {}

for layer in new_model.layers:
  class_of_layer[layer_name(layer)] = layer.__class__.__name__


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
      if activation == "linear":
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
dprint("params_of_activation:"+pretty(params_of_activation))
dprint("activation_of_layer:"+pretty(activation_of_layer))

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
      inbound = inbound_by_name[layer_name(layer)][0]
      inbound_layer = model.get_layer(inbound)
      if type(inbound_layer) == Conv2D:
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

replaced_layers = replaced_activation_layers(orig_inbound_by_name, new_model)
dprint("replaced_layers:"+str(replaced_layers))
dprint()

# key-value reversed version of replaced_layers
original_activation_layer_of = {v:k for k,v in replaced_layers.items()}
dprint("original_activation_layer_of:"+str(original_activation_layer_of))

inbound_by_name = collapsed_layers(orig_inbound_by_name, replaced_layers)
dprint("inbound_by_name:")
dprint(pretty(inbound_by_name))
dprint()

outbound_by_name = outbound_layers(inbound_by_name)
dprint("outbound_by_name:")
dprint(pretty(outbound_by_name))
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

raw_sections = []
for section in section_dict.values():
  raw_sections.append(section)

dddprint(pretty(raw_sections))

# returns a list of all sections that will have to be defined
# as referenceable constants
def defined_sections(sections):
  result = []
  for section in sections:
    result.append(section.name)
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
  referencing = referencing_sections(raw_sections)

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
dprint("Sections ordered properly by referencing:")
dprint(pretty(sections))

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
    dprint(str(self.params))
    line = "let {} = {}(".format(self.name, self.forge_class)
    i = 1
    for swift_param in self.params:
      origin = self.params[swift_param]
      dprint("swift_param:'"+swift_param+"', origin:'"+str(origin)+"'")
      cfg_name = origin[0] # name of the keras config item
      # third argument is a precomputed value if first argument is ""
      # or a conversion function if present
      converter = to_swift if len(origin) < 3 or not is_function(origin[2]) else origin[2]
      val = converter(self.cfg[cfg_name]) if cfg_name != "" else origin[2]
      # do not emit code if actual value is default value
      default_val = origin[1] if len(origin) > 1 else None
      if not default_val or to_swift(default_val) != val:
        arg = "{}: {}".format(swift_param, val)
        if i > 1:
          line += ", "
        line += arg 
        i += 1
    line += ")"
    dprint(line)
    return [line]    

# Generate Forge class "Convolution":
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
      ("kernel"    , ("kernel_size",)),
      ("channels"  , ("filters",)),
      ("stride"    , ("strides", (1, 1))),
      ("padding"   , ("padding", ".same", to_swift_enum)),
      ("useBias"   , ("use_bias", True)),
      ("activation", ("", "nil", 
        var_name_of_activation[activation_of_layer[activation_layer(layer.name)]])),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Convolution", params)

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

# Generate Forge class "Dense":
#
# init(neurons: Int,
#   activation: MPSCNNNeuron? = nil,
#   useBias: Bool = true,
#   name: String = "")
class ForgeDense(ForgeLayer):
  def __init__(self, layer):
    params = OrderedDict([ 
      ("neurons" , ("units",)),
      ("useBias" , ("use_bias", True)),
      ("name"    , ("name","",quote_string))
    ])
    super().__init__(layer, "Dense", params)

# Forge supports only one input so far
input_already_defined = False
def inputSource(width, height, name):
  if input_already_defined:
    raise RuntimeError("Forge supports only one input but network has multiple inputs")
  src = []
  src.append("let input = Input()")
  # inception V3 wants input scaled between -1 and 1, but we can not
  # know that from the model; there is nothing about that in the input config
  if MODEL == "INCEPTION_V3":
    src.append("let {} = input --> Resize(width: {}, height: {}) --> Activation(input_scale)"\
                  .format(name, width, height))
  else:
    src.append("let {} = input --> Resize(width: {}, height: {})"\
                  .format(name, width, height))
  dprint(src)
  input_defined = True
  return src

def inputSourceFromConfig(layer):
  cfg = layer.get_config()
  dprint(pretty(cfg))
  return inputSource(cfg["batch_input_shape"][1], cfg["batch_input_shape"][2], layer_name(layer))

# start swift source generation

swift_src = []
swift_src.append("")
swift_src.append("// begin of autogenerated forge net generation code")
swift_src.append("")
swift_src.append("var model:Model")
swift_src.append("")

# declare activation functions

used_prefix_counter = {}
var_name_of_activation = {}

for activation in activations:
  params = params_of_activation[activation]
  
  if 'swift_prefix' in params and params['swift_prefix']:
    prefix = params['swift_prefix']

    if prefix in used_prefix_counter.keys():
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
if MODEL == "INCEPTION_V3":
  swift_src.append("let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)")

for index, layer in enumerate(new_model.layers):
  name = layer_name(layer)
 
  dprint(str(index)+":"+layer.__class__.__name__+" name:"+name)
  dprint(str(index)+"type::"+str(type(layer)))
  
  inbound_layers = orig_input_layers(inbounds, new_model)

  if layer_name(layer) in replaced_layers.keys():
    dprint("Layer '"+name+"' has been replaced by activation in layer '"+replaced_layers[name])
  else:
    if type(layer) == Conv2D:
      swift_src.extend(ForgeConv2D(layer, var_name_of_activation).swift_source())
    elif type(layer) == keras.engine.topology.InputLayer:
      swift_src.extend(inputSourceFromConfig(layer))
    elif type(layer) == MaxPooling2D:
      swift_src.extend(ForgeMaxPooling2D(layer).swift_source())
    elif type(layer) == AveragePooling2D:
      swift_src.extend(ForgeAveragePooling2D(layer).swift_source())
    elif type(layer) == GlobalAveragePooling2D:
      swift_src.extend(ForgeGlobalAveragePooling2D(layer).swift_source())
    elif type(layer) == Dense:
      swift_src.extend(ForgeDense(layer).swift_source())
    elif layer.__class__.__name__ == "Concatenate":
      dprint("Concatenate layer '"+name+"' will not be predefined here")
    elif type(layer) == Activation:
      swift_src.append(activationSourceFromConfig(layer))
    elif type(layer) == keras.layers.core.Lambda and layer.name == "space_to_depth_x2":
      swift_src.append('let {} = SpaceToDepthX2(name: "{}")'.format(layer.name, layer.name))
    else:
      raise ValueError("Unknown layer type:"+str(type(layer))+"\nConfig:\n"+
                       pretty(layer.get_config()))


swift_src.append("")

swift_src.append("do {")

for section in sections:
  var_name = section.name
  if isinstance(section, ChainSection):
    line = "let "+var_name + " = "
    char_offset = len(line)
    for index, layer_id in enumerate(section.layers):
      line += layer_id
      if len(line)-char_offset > 70:
        line +="\n        "
        char_offset += len(line)
      if index < len(section.layers)-1:
        line += " --> "
    swift_src.append(line)
  elif isinstance(section, ConcatSection):
    line = "let "+var_name + " = Concatenate(["
    for index, layer_id in enumerate(section.layers):
      line += layer_id
      if index < len(section.layers)-1:
        line += ", "
    line += "])"
    swift_src.append(line)
  else:
    raise RuntimeError("Unknown section type")

# This is a bit hacky, we should always insert a Softmax layer after a
# Dense keras layer with softmax activatio (not just for this model)
if MODEL == "INCEPTION_V3":
  swift_src.append("let output = {} --> Softmax()".format(output_layers[0]))
else:
  swift_src.append("let output = {}".format(output_layers[0]))
swift_src.append("model = Model(input: input, output: output)")
swift_src.append("}")
swift_src.append("let success = model.compile(device: device, inflightBuffers: inflightBuffers) { ")
swift_src.append("  name, count, type in ParameterLoaderBundle(name: name,")
swift_src.append("  count: count,")
swift_src.append('  prefix: "{}-",'.format(model_name))
swift_src.append('  suffix: type == .weights ? ".weights" : ".biases",')
swift_src.append('  ext: "bin")')
swift_src.append("}")
swift_src.append("")
swift_src.append("// end of autogenerated forge net generation code")

for line in swift_src:
  print(line)

new_model.save(file_name_plus(model_path, "_nobn"))

'''
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
'''

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
#if DO_STATIC_CONVERSION:
#  features_nobn = model_nobn.predict(image_data)
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

#if DO_STATIC_CONVERSION:
#  compare_features(features, features_nobn)
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



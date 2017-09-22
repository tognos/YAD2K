import h5py
import numpy as np
import json

#filename = 'model_data/yolo_nobn.h5'
#outname = 'model_data/yolo_nobn.fp16.h5'
#filename = 'model_data/resnet50_nobn.h5'
#outname = 'model_data/resnet50_nobn.fp16.h5'
filename = 'model_data/vgg_16_nobn.h5'
outname = 'model_data/vgg_16_nobn.fp16.h5'

f = h5py.File(filename, 'r')
out_f = h5py.File(outname, 'w')

def replace_in_dict(node, which_key, what, with_what):
  #print("Node: class:", node.__class__)
  if node.__class__.__name__ == "dict":
    for key, value in node.items():
      #print("Checking key'",key, "', value class:", value.__class__)
      if value.__class__.__name__ == "dict" or value.__class__.__name__ == "list":
        replace_in_dict(value, which_key, what, with_what)
      elif key == which_key and value == what:
        print("Replacing",which_key,"with",with_what,"for key",key)
        node[key] = with_what
  elif node.__class__.__name__ == "list":
    for item in node:
       replace_in_dict(item, which_key, what, with_what)
 
for key, value in f.attrs.items():
  if key == "model_config":
    #print("converting file attr :",key," data:",value, " type:",value.__class__)
    model_config = json.loads(value.decode('utf-8'))
    replace_in_dict(model_config, "dtype", "float32", "float16")
    new_value = json.dumps(model_config).encode('utf-8')
    #print(new_value.__class__.__name__)
    print("adding converted file attr:",key," data:",new_value)
    out_f.attrs.create(key, new_value)
  else:
    print("adding file attr name:",key," data:",value, " type:",value.__class__)
    out_f.attrs.create(key, value)

def convert(name,obj):
  print("Converting",name)
  #print(obj, "attr:", obj.attrs)

  #print(obj.__class__.__name__)
  if obj.__class__.__name__ == 'Group':
    print("Create group:",name)
    new_item = out_f.create_group(name)
  elif obj.__class__.__name__ == 'Dataset':
    print("Create dataset:",name)
    new_item = out_f.create_dataset(name, data=obj, shape = obj.shape, dtype=np.float16)
  else:
    print("ERROR: unknown class")

  for key, value in obj.attrs.items():
    print("adding attr name:",key," data:",value)
    new_item.attrs.create(key, value)

  #print()

def show(name,obj):

  #print(obj.__class__.__name__)
  if obj.__class__.__name__ == 'Group':
    print("Group:",name)
  elif obj.__class__.__name__ == 'Dataset':
    print("Dataset:",obj)
  else:
    print("ERROR: unknown class")

  for key, value in obj.attrs.items():
    print("has attr name:",key," data:",value)

  #print()

f.visititems(show)
print("-----> Converting:")
f.visititems(convert)
print("-----> New:")

out_f.visititems(show)

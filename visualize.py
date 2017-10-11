#! /usr/bin/env python
"""compare float array files."""
import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description='compare .float binary files')
parser.add_argument('dir1', help='path to directory containing .float files')
parser.add_argument('dir2',  nargs='?', help='path to another directory containing .float files')
parser.add_argument('--shapes1', default="shapes.json", help='name of .json file with shapes in dir1')
parser.add_argument('--shapes2', default="shapes.json", help='name of .json file with shapes in dir2')

def compare_features_fast(features1, features2):
  error = np.abs(features1 - features2) 
  max_error = np.max(error)
  avrg_error = np.sum(error)/features1.size
  print("error max:",max_error,"avrg:",avrg_error)

def vis_square(data):
  """Take an array of shape (n, height, width) or (n, height, width, 3)
  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
  
  # normalize data for display
  data = (data - data.min()) / (data.max() - data.min())
  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = (((0, n ** 2 - data.shape[0]),
        (0, 1), (0, 1))                 # add some space between filters
      + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
  data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
  plt.imshow(data); plt.axis('off')
  print("visualized")


def stat_string(data):
  return "Min: {} Max: {} Avrg: {}".format(np.min(data), np.max(data), np.sum(data)/data.size)

def visualize1(data1):
  print("visualizing")
  fig = plt.figure()
  vis_square(data1)
  ttext = plt.title(stat_string(data1))
  plt.show()


def visualize2(data1, data2):
  print("visualizing")
  fig = plt.figure()
  a=fig.add_subplot(1,2,1)
  vis_square(data1)
  ttext = plt.title(stat_string(data1))
  a=fig.add_subplot(1,2,2)
  vis_square(data2)
  ttext = plt.title(stat_string(data2))
  plt.setp(ttext, size='medium', name='helvetica', weight='light')
  plt.show()

def visualize3(data1, data2):
  print("visualizing")
  data3 = np.abs(data2 - data1)
  stat1 = stat_string(data1)
  stat2 = stat_string(data2)
  stat3 = stat_string(data3)
  fig = plt.figure()
  a=fig.add_subplot(1,3,1)
  vis_square(data1)
  plt.setp(plt.title(stat1), size='medium', name='helvetica', weight='light')
  a=fig.add_subplot(1,3,2)
  vis_square(data2)
  plt.setp(plt.title(stat2), size='medium', name='helvetica', weight='light')
  a=fig.add_subplot(1,3,3)
  vis_square(data3)
  plt.setp(plt.title(stat3), size='medium', name='helvetica', weight='light')
  plt.show()


def shape_for_path(path_name, shapes):
  base = os.path.basename(path_name)
  return shapes[base]

def check2(file1, file2, shapes1, shapes2):
  print("Checking "+file1+" and "+file2)
  data1 = np.fromfile(file1, dtype = np.float32)
  data2 = np.fromfile(file2, dtype = np.float32)
  shape1 = shape_for_path(file1, shapes1)
  shape2 = shape_for_path(file2, shapes2)
  if -1 in shape2 or len(shape2) < 3:
    print("shape2 is undefined or flat, using shape1 for both inputs")
    shape2 = shape1
  elif -1 in shape1 or len(shape1) < 3:
    print("shape1 is undefined or flat, using shape2 for both inputs")
    shape1 = shape2

  print("shape1:", shape1)
  print("shape2:", shape2)
  width, height, channels, images = shape1
  if channels == 3:
   # 3 channels, make color image
    axes = (3,0,1,2)
    data1 = data1.reshape(shape1).transpose(axes)
    data2 = data2.reshape(shape2).transpose(axes)
    data1 = data1[:, :, :, ::-1]
    data2 = data2[:, :, :, ::-1]
    print("vis shape1:", data1.shape)
    print("vis shape2:", data2.shape)
    # Take an array of shape (n, height, width, 3)
  else:
    # n channels, show n single channel images
    axes = (2,1,0)
    data1 = data1.reshape(shape1[0:3]).transpose(axes)
    data2 = data2.reshape(shape2[0:3]).transpose(axes)
    print("vis shape1:", data1.shape)
    print("vis shape2:", data2.shape)
    # Take an array of shape (n, height, width) or (n, height, width, 3)
  visualize3(data1, data2)
  exit(0)#

def check1(file1, shapes1):
  print("Checking "+file1)
  data1 = np.fromfile(file1, dtype = np.float32)
  shape1 = shape_for_path(file1, shapes1)
  print("orig shape1:", shape1)
  width, height, channels, images = shape1
  if channels == 3:
    # 3 channels, make color image
    axes = (3,0,1,2)
    data1 = data1.reshape(shape1).transpose(axes)
    print("vis shape1:", data1.shape)
    # Take an array of shape (n, height, width, 3)
  else:
    # n channels, show n single channel images
    axes = (2,1,0)
    data1 = data1.reshape(shape1[0:3]).transpose(axes)
    print("vis shape1:", data1.shape)
    # Take an array of shape (n, height, width) or (n, height, width, 3)
  visualize1(data1)
  exit(0)


def load_json(file_name):
    print("Loading "+file_name)
    with open(file_name) as json_data:
      data = json.load(json_data)
    return data

def load_shapes(data_path, shape_file):
  data_dir = os.path.dirname(data_path)
  shape_path = os.path.join(data_dir, shape_file)
  if os.path.exists(shape_path):
    return load_json(shape_path)
  return None

def _main(args):
    dir1 = os.path.expanduser(args.dir1)
    shapes1 = load_shapes(dir1, args.shapes1)

    if args.dir2 is None:
      check1(dir1, shapes1)
      exit(0)

    dir2 = os.path.expanduser(args.dir2)
    #print(shapes1)
    shapes2 = load_shapes(dir2, args.shapes2)
    #print(shapes2)

    if os.path.isfile(dir1) and os.path.isfile(dir2):
      check2(dir1, dir2, shapes1, shapes2)
    else:
      dir1_files = glob.glob(os.path.join(dir1,"*.floats"))
      dir2_files = glob.glob(os.path.join(dir2,"*.floats"))

      #print(dir1, dir1_files)
      #print(dir2, dir2_files)

      dir1_names = {}
      for path in dir1_files:
        dir1_names[os.path.basename(path)] = path
        
      dir2_names = {}
      for path in dir2_files:
        base = os.path.basename(path)
        if base in dir1_names:
          check(dir1_names[base], path)
      
if __name__ == '__main__':
      _main(parser.parse_args())

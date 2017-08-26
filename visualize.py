#! /usr/bin/env python
"""compare float array files."""
import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='compare .float binary files')
parser.add_argument('dir1', help='path to directory containing .float files')
parser.add_argument('dir2', help='path to another directory containing .float files')


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

def compare_features_lin(features1, features2):
  max_error = 0
  for i in range(features1.shape[0]):
        diff = np.abs(features1[i] - features2[i])
        max_error = max(max_error, diff)
        if diff > 2:
          print(i, ":", features1[i], features2[i], diff)
  print("Largest error:", max_error)

def compare_features_best(features1, features2):
  for i in range(features1.shape[0]):
    min_error = 1e20
    for j in range(features1.shape[0]):
        diff = np.abs(features1[i] - features2[j])
        if diff < min_error:
          min_error = diff
          min_j = j
    print(i,"->",min_j, " min_error=",min_error)

def compare_features_min(features1, features2):
  max_error = 0
  for i in range(features1.shape[0]):
        diff = np.abs(features1[i] - features2[i])
        if diff < 1e-3:
          print(i,"-> error=",diff)

def compare_features_fast(features1, features2):
  error = np.abs(features1 - features2) 
  max_error = np.max(error)
  avrg_error = np.sum(error)/features1.size
  print("error max:",max_error,"avrg:",avrg_error)


def compare_features_(features1, features2):
  for i in range(features1.shape[0]):
    min_error = 1e20
    for j in range(features1.shape[0]):
        diff = np.abs(features1[i] - features2[j])
        if diff < min_error:
          min_error = diff
          min_j = j
    print(i,"->",min_j, " min_error=",min_error)

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

def visualize(data):
  print("visualizing")
  data = data.reshape(425,19,19)
  print("data:", data.shape)
  vis_square(data)
  plt.show()

def visualize2(data1, data2):
  print("visualizing")
  fig = plt.figure()
  a=fig.add_subplot(1,2,1)
  vis_square(data1)
  a=fig.add_subplot(1,2,2)
  vis_square(data2)
  plt.show()


def check(file1, file2):
  print("Checking "+file1+" and "+file2)
  data1 = np.fromfile(file1, dtype = np.float32)
  data2 = np.fromfile(file2, dtype = np.float32)
  visualize2(data1.reshape(425,19,19),data2.reshape(425,19,19))
  #visualize2(data1.reshape(32,608,608),data2.reshape(32,608,608))
  exit(0)
  #compare_features_min(data1, data2)
  compare_features_fast(data1, data2)

def _main(args):
    dir1 = os.path.expanduser(args.dir1)
    dir2 = os.path.expanduser(args.dir2)

    if os.path.isfile(dir1) and os.path.isfile(dir2):
      check(dir1, dir2)
    else:
      dir1_files = glob.glob(os.path.join(dir1,"*.floats"))
      dir2_files = glob.glob(os.path.join(dir2,"*.floats"))

      print(dir1, dir1_files)
      print(dir2, dir2_files)

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

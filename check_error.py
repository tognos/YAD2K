#! /usr/bin/env python
"""compare float array files."""
import argparse
import os
import numpy as np
import glob

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
  min_1 = np.min(features1)
  min_2 = np.min(features2)
  max_1 = np.max(features1)
  max_2 = np.max(features2)
  avrg_1 = np.sum(features1)/features1.size
  avrg_2 = np.sum(features2)/features2.size
  print("error max:",max_error,"avrg:",avrg_error)
  print("avrg1:",avrg_1,"min1:",min_1,"max1:", max_1)
  print("avrg2:",avrg_2,"min2:",min_2,"max2:",max_2)


def compare_features_(features1, features2):
  for i in range(features1.shape[0]):
    min_error = 1e20
    for j in range(features1.shape[0]):
        diff = np.abs(features1[i] - features2[j])
        if diff < min_error:
          min_error = diff
          min_j = j
    print(i,"->",min_j, " min_error=",min_error)



def check(file1, file2):
  print("Checking "+file1+" and "+file2)
  data1 = np.fromfile(file1, dtype = np.float32)
  data2 = np.fromfile(file2, dtype = np.float32)
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

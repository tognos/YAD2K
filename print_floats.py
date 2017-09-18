#! /usr/bin/env python
"""compare float array files."""
import argparse
import os
import numpy as np
import glob
import json

def pretty(the_dict):
  return json.dumps(the_dict,sort_keys=True, indent=4)

def print_one_per_line(data):
  for i in range(data.size):
    print(data[i])

parser = argparse.ArgumentParser(description='print .float binary files')
parser.add_argument('input', help='path to .float file')
def _main(args):
    input_name = os.path.expanduser(args.input)
    data = np.fromfile(input_name, dtype = np.float32)
    np.set_printoptions(threshold=np.inf)
    #print(data)
    print_one_per_line(data)
    

if __name__ == '__main__':
      _main(parser.parse_args())

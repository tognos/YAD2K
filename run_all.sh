#!/bin/bash

set -e

models=("TINY_YOLO" "YOLO" "INCEPTION_V3" "RESNET_50" "VGG16" "INCEPTION_RESNET_V2" "MOBILE_NET" "XCEPTION")

for model in "${models[@]}"
do
  ./keras2metal.py $model "$@"
done

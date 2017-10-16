# YAD2K: Yet Another Darknet 2 Keras 2 Apple Metal and Forge

This repo is a fork of YADK2 and incorporates its functionality, but goes way beyond and allows all Keras standard models to be converted to the iOS Forge framework, or more precisely, an [extended fork of the Forge framework](https://github.com/tognos/Forge).

This combination allows to convert the weights and layers of the following models out of the box:

* Tiny Yolo
* Yolo 2
* VGG16
* Resnet-50
* Inception-V3
* InceptionResnet-V2
* Keras Mobilenet
* Xception

The weight will be exported as raw binary files with a layout suitable for MPSCNN, and the network is exported as .swift source code for the Forge framework that will create and run a model in Forge on iOS.

The converter does the following optimizations:

* collapses batchnorm layers by multiplying their parameters onto previous layer weigths
* turns activation layers into MPSCNNConvolution activations
* arranges weights properly so that flatten or reshape layers are not needed
* Layer concatenation and combining multiple inputs for residual connection is done in Forge-Style with minimal or no additional copying of data

There are also python scripts included for the following purposes:

* `test_classifier.py` runs predictions on images and writes all intermediate feature maps to disk
* `visualize.py` compares the feature maps with feature maps produced by Forge and allows to quickly pinpoint where the Forge model diverges from the Keras model
* `convertfp16.py` converts any keras model from float32 to float16; the hope was to get the same feature maps as in iOS with 16-Bit MPSCNN, but this is not the case; the average error between iOS and Keras produced feature maps is about the same regardless of the Keras precision and is typically in the order of 10e-3 and 10e-5.
* `check_error.py` does a numerical comparison between feature maps in different directories

## keras2metal.py

The centerpiece is the `keras2metal.py` conversion script. It started as some modifications of the yolo2metal.py that comes with Forge and became a ~1800 lines long piece of code that is not exactly a pinnacle of software engineering, but it is quite generic and does not contain much model specific code, so adding a new model takes from a few minutes up to a day or two if there are some layers in it not supported out of the box and need to be implemented or composed on the Forge side.

The script does the following:

* create an original model using the keras.applications namespace or load a .h5 model from disk (yolo)
* performs batchnorm folding and writes this simplified keras model as .h5 to disk (named `model_name_nobn.h5`)
* writes the layer weights in Metal-suitable format to disk
* creates .swift source code for the Forge framework that loads the weights and creates and runs inference on this net

Optionally:

* it plots dot-graphs of the original and simplified models
* runs a classifier on the original and the optimized model and prints the result
* writes the weights of the original (non-optimized) model to disk

The full usage is here:

```
usage: keras2metal.py [-h] [--param_dir PARAM_DIR]
                      [--orig_param_dir ORIG_PARAM_DIR]
                      [--model_dir MODEL_DIR] [--src_out_dir SRC_OUT_DIR]
                      [--image_dir IMAGE_DIR] [-p] [-v] [-d] [-r] [-s] [-e]
                      [-a] [-q]
                      {TINY_YOLO,YOLO,INCEPTION_V3,RESNET_50,VGG16,INCEPTION_RESNET_V2,MOBILE_NET,XCEPTION}

Convert Keras Models to Forge Metal Models

positional arguments:
  {TINY_YOLO,YOLO,INCEPTION_V3,RESNET_50,VGG16,INCEPTION_RESNET_V2,MOBILE_NET,XCEPTION}

optional arguments:
  -h, --help            show this help message and exit
  --param_dir PARAM_DIR
                        Path to directory to write the metal weights for
                        optimized models (default: Parameters)
  --orig_param_dir ORIG_PARAM_DIR
                        Path to directory to write the metal weights of non-
                        optimized models (default: OriginalParameters)
  --model_dir MODEL_DIR
                        Path to directory where keras and yolo models are
                        stored (default: model_data)
  --src_out_dir SRC_OUT_DIR
                        Path to directory where the swift source code will be
                        written (default: generated_src)
  --image_dir IMAGE_DIR
                        Path to directory where images for classification
                        tests are to be found (default: images/classify)
  -p, --plot_models     Plot the models and save as image. (default: False)
  -v, --verbose         be verbose about what the converter is doing (default:
                        False)
  -d, --debug           be very verbose about what the converter is doing
                        (default: False)
  -r, --run_classifier  run a classification models on test input (default:
                        False)
  -s, --save_orig_models
                        save also the non-optimized keras models as .h5
                        (default: False)
  -e, --export_orig_weigths
                        export also the non-optimized model weigths (default:
                        False)
  -a, --keep-activations
                        keep separate activation layers and do not put them as
                        forge conv layer activations (default: False)
  -q, --quick_run       skip optimzed model generation and weight export
                        reusing model from previous run (default: False)
```

## Quick Start
Install the Requirements below; if you want to plot models, you need py-dot and graphviz as well, and it is highly recommended to use model graphs when dealing with complex networks.

* Check out the [forked Forge Framework for iOS](https://github.com/tognos/Forge)
* if you want to convert the YOLO nets as well, perform the steps below under "Welcome to YAD2K"
* try to run `./keras2metal MOBILE_NET -v -p`
* fix any dependency problems you might have
* to generate all available models, run the `./run_all.sh` script
* copy weights from the subdirectory "Parameters" and swift code from "generated_src" to the Forge project folder; when YADK and Forge reside in the same directory, the commands would be

```
$ cp Parameters/* ../Forge/ForgeTests/ForgeTests/TestWeights/
$ cp generated_src/* ../Forge/Forge/Forge/Networks/
```

You can also direct the script to write parameters directly to these directories:
```
./keras2metal MOBILE_NET --param_dir ../Forge/ForgeTests/ForgeTests/TestWeights/ --src_out_dir ../Forge/Forge/Forge/Networks/
```

Use -v or -d if you are bored to see what's going on.

Options given to the` ./run_all.sh` script are passed to `./keras2metal.py`

## Workflow to add conversion for a new network

* Edit `keras2metal.py` and add a new `if MODEL_NAME == ""` section; don't forget to add it to the argument parser
* run `keras2metal.py` and see if it breaks and where
* The most probable causes for new models not to work are unknown activations or layers
* To emit source code for new layers, derive a class from `ForgeLayer` and add it to the "if"-cascade where it bailed
* To emit source code for new activations, find the corresponding MPSCNNNeuron type and parameters and extend the `gather_activations()` function
* When you finally have the script running and producing weights and source code, add both to Forge and add a test to `ModelTests.swift`
* The swift compiler will also help you to ensure the code you generated is sound; there should be no warnings in the generated code
* Call the test function on the device with the parameter debug: true; it will write out feature maps for every layer to the document directory on the device
* Copy the .dot output from the Xcode console into a file and convert to a png or whatever you want to look at; compare it to the keras model graph created by running `keras2metal.py` with the option `-p`

* Extend `test_classifier.py` to handle your model; use the `<model_name>_nobn.h5` to produce feature maps from the keras model
* Use iTunes to copy the feature maps from the device onto your computer into a directory `~/Downloads/Features`

* You probably need to install pyenv or virtualenv to switch between the system python and you keras/python3 installation because only the system python can open a window

* Use `visualize.py Features/<netname>-<layername>.floats ~/Downloads/Features/<netname>-<layername>.floats --shapes2 shapes-<netname>.json`
to compare the feature maps; take a look at the graph images to identify corresponding layers; in Forge, the conv layers typically contain the activation, in Keras you often have separate activation layers, so you need to compare the Forge activations of the conv layers with the output of the corresponding Keras activation layer

* Start comparing the input images; in the Keras features you will see what kind of preprocessing has been applied; typically the channels might be reversed, the values need to be scaled, and sometimes the mean is subtracted

* If necessary, fix the conversion script to produce the proper kind of preprocessing code; you can set some variables that will make the script to produce all kinds of preprossing code:

	* `SWAP_INPUT_IMAGE_CHANNELS`: when true, code for input channel swap RGB->BGR will be generated
	* `SUBTRACT_IMAGENET_MEAN` : when true, code for imagenet mean subtraction will be generated 
	* `SCALE_INPUT_TO_MINUS_1_AND_1`: when true, the input image values will be scaled to -1 .. 1`
	 	
* if you need more or other preprocessing, combine the settings or write a custom kernel for it

* When the input image on the Forge side is basically the same like the one on the keras side, you will still have artifacts from different scaling methods, so run test_classifier.py with the preprocessed output feature file you got from the device; it ends in .float

* When you compare the inputs now, you should have two layers where the error is exactly 0.0

* Now advance down through the network and compare more outputs; the avarage error should not be larger 0.01 and be typically much smaller

* if the average error is larger, look at the range of the values; only when they are in the hundreds or thousands the error is allowed to be larger than e.g. 0.1.

* When you find a larger error, the reasons can be numerous. I encountered the following:
	* Weigths not in proper order: This has not happened for a while now, but was a major PITA during making things work, and the converter now does a lot of magic reshaping and transposing weight so the Metal shaders are happy, especially when it comes to format the weights properly for a Dense layer with leaving out reshape and flatten layers, so the format of the weight for the Dense layer depends on the kernel format of previuos layers. Also note that some Layers like the Keras SeparatedConvolution has two sets of weigths + bias that just come as a list; activating the debug output (-d) will tell you a lot about what the converter thinks it is doing there
	* Wrong activations or activation parameters; they are usually easy to find and fix
	* Special combinations of stride and padding; there was e.g. a problem with kernels of size 1 and larger strides, but this is fixed
	* Bugs in MPSCNN: e.g. the ReLUN activation does not work in a convolution layer, only as separate activation layer; you can run the script with the -a option to keep seperate activation layers
	* Bugs in Forge: There might be some topologies or special combinations of layers that might cause trouble, especially when the net is branching out or merging; in these cases some custom layers may cause trouble when they do not correctly handle read and write offsets in conjunction with Concat or Collect tensors; most should work now; you can check if the custom layers are aware of destinationChannnelOffset and destinationImageNumber and MPSOffet or src image numbers; if not, that will be a likely cause; the solution is either to rearrange the network if possible or fix the problem in the Forge network; however, the current selection of networks already contain a lot of different combinations
	* Do not load very large input images into Forge; they will be loaded as a full size Metal texture into the texture memory before being resized; it will silently have bad side effects causing undefined behavior; 1k x 1k should work, but I ran into trouble with e.g. 2738x1825 images.
	* Bugs in Metal/MPS: Some stuff does not work as advertised in the documention; e.g. I had strange issues combining blit encoders with MPS; the Space2Depth layer required for YOLO works fine with a blit encoder, but I could not make the ZeroPadding2D-Layer reliably work with a blit encoder so there is a custom shader there for it now.
* However, despite all above potential for trouble, I am confident most convolutional Keras models can be made to work Forge with reasonable effort now; a lot of the troublesome stuff is already solved, and this workflow makes it really quick and easy to find what goes wrong even when it has over hundreds of layers like the InceptionResnet_V2, which converts and runs fine on the iPhone7

## Performance of the converted networks

On iPhone 7 and 7 Plus:

| Net                      | fps  | top1  | top5  | layers | tensors | parameters  |
|--------------------------|------|-------|-------|--------|---------|-------------|
| TinyYolo  (VOC, 416x416) |   15 |     - |     - |     16 |      17 |  15.858.717 |
| Yolo 2 (COCO, 608x608)   |  2.2 |     - |     - |     30 |      32 |  50.952.553 |
| VGG16                    |  7.2 | 0.715 | 0.901 |     26 |      27 | 138.357.544 |
| Resnet-50                | 21.3 | 0.759 | 0.929 |     93 |     110 |  25.530.472 |
| Inception-V3             | 17.4 | 0.788 | 0.944 |    112 |     124 |  23.817.352 |
| InceptionResnet-V2       |  5.8 | 0.804 | 0.953 |    373 |     457 |  55.813.192 |
| Mobilenet (*)            | 30.5 | 0.665 | 0.871 |     59 |      60 |   4.221.032 |
| Xception                 |  5.1 | 0.790 | 0.945 |    106 |     119 |  22.828.688 |

(*) With separate relu6 activation layers because of a MPSCNNConvolution bug

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Welcome to YAD2K (YOLO/Darknet instructions)

You only look once, but you reimplement neural nets over and over again.

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

![YOLO_v2 COCO model with test_yolo defaults](etc/dog_small.jpg)

--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [pydot-ng](https://github.com/pydot/pydot-ng) (Optional for plotting model.)

### Installation
```bash
git clone https://github.com/allanzelener/yad2k.git
cd yad2k

# [Option 1] To replicate the conda environment:
conda env create -f environment.yml
source activate yad2k
# [Option 2] Install everything globaly.
pip install numpy h5py pillow
pip install tensorflow-gpu  # CPU-only: conda install -c conda-forge tensorflow
pip install keras # Possibly older release: conda install keras
```

## Quick Start

- Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
- Convert the Darknet YOLO_v2 model to a Keras model.
- Test the converted model on the small test set in `images/`.

```bash
wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
./test_yolo.py model_data/yolo.h5  # output in images/out/
```

See `./yad2k.py --help` and `./test_yolo.py --help` for more options.

--------------------------------------------------------------------------------

## More Details

The YAD2K converter currently only supports YOLO_v2 style models, this include the following configurations: `darknet19_448`, `tiny-yolo-voc`, `yolo-voc`, and `yolo`.

`yad2k.py -p` will produce a plot of the generated Keras model. For example see [yolo.png](etc/yolo.png).

YAD2K assumes the Keras backend is Tensorflow. In particular for YOLO_v2 models with a passthrough layer, YAD2K uses `tf.space_to_depth` to implement the passthrough layer. The evaluation script also directly uses Tensorflow tensors and uses `tf.non_max_suppression` for the final output.

`voc_conversion_scripts` contains two scripts for converting the Pascal VOC image dataset with XML annotations to either HDF5 or TFRecords format for easier training with Keras or Tensorflow.

`yad2k/models` contains reference implementations of Darknet-19 and YOLO_v2.

`train_overfit` is a sample training script that overfits a YOLO_v2 model to a single image from the Pascal VOC dataset.

## Known Issues and TODOs

- Expand sample training script to train YOLO_v2 reference model on full dataset.
- Support for additional Darknet layer types.
- Tuck away the Tensorflow dependencies with Keras wrappers where possible.
- YOLO_v2 model does not support fully convolutional mode. Current implementation assumes 1:1 aspect ratio images.

## Darknets of Yore

YAD2K stands on the shoulders of giants.

- :fire: [Darknet](https://github.com/pjreddie/darknet) :fire:
- [Darknet.Keras](https://github.com/sunshineatnoon/Darknet.keras) - The original D2K for YOLO_v1.
- [Darkflow](https://github.com/thtrieu/darkflow) - Darknet directly to Tensorflow.
- [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo) - YOLO_v1 to Caffe.
- [yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) - YOLO_v2 in PyTorch.

--------------------------------------------------------------------------------

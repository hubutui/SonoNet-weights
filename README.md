# SonoNet weights

This repository contains pretrained weights and model descriptions for all of
the SonoNet variations described in our recent submission:

Baumgartner et al., ["Real-Time Detection and Localisation of Fetal Standard Scan Planes in 2D
Freehand Ultrasound"](https://arxiv.org/abs/1612.05601), arXiv preprint:1612.05601 (2016),

and prior work published here:

Baumgartner et al., ["Real-time standard scan plane detection and localisation in fetal ultrasound
using fully convolutional neural networks"](http://link.springer.com/chapter/10.1007/978-3-319-46723-8_24), Proc. MICCAI (2016).

Please acknowledge the first of the two papers above if you end up using the
these weights for your work.

The networks were trained using [theano](http://deeplearning.net/software/theano/)
and the [lasagne](https://github.com/Lasagne/Lasagne) deep learning framework.

The weights are saved in the respective `.npz` files, the model definitions are
given in `models.py`. A minimal example for classifying the images in the folder
`example_images` is given in `example.py`.

## Setup

Running `example.py` requires the latest theano and lasagne versions. Follow
the instructions [here](http://lasagne.readthedocs.io/en/latest/user/installation.html),
under the section "Bleeding Edge".

Furthermore, `numpy`, `scipy` are required.

Once everything is set up you can simply run:

    python example.py 

## Demo videos

Demo videos are available on Youtube:

[![Youtube video 1](http://img.youtube.com/vi/yPCvAdOYncQ/0.jpg)](http://www.youtube.com/watch?v=yPCvAdOYncQ)
[![Youtube video 2](http://img.youtube.com/vi/4V8V0jF0zFc/0.jpg)](http://www.youtube.com/watch?v=4V8V0jF0zFc)


# Update 2018-10-02 by Butui Hu

To run `example.py`, you need the latest [Theano](https://github.com/Theano/Theano), the latest [Lasagne](https://github.com/Lasagne/Lasagne), old version [SciPy](https://www.scipy.org/)(version 0.17 works ok) with Python2.

`to_pytorch_model.py` helps convert Lasagne model to PyTorch model, PyTorch model is defined in `SonoNet.py`, you need PyTorch >= 0.4.1 with Python3. Converted PyTorch models is provided in `pytorch.model.tar.gz` for convenient.

`crop_image.py` helps crop example image according to `example.py`, and this requires [imageio](http://imageio.github.io/) module.

**Note**: `example_images/test/other/other.jpg` is not provided by paper author, and is not cropped yet. Please also note that though I try my best to convert Lasagne model to PyTorch model, I could not make sure that the model works as expected in Lasagne. And after I write a simple script to run the example, I didn't get the same performance. 

Saddly, I have to give up. The model is trained with grayscale image, and data is normalize to [0, 255] not [0, 1]. I don't think that I could use these pretrained model.


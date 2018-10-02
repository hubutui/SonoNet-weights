#!/bin/env python3
#
# don't know how to do with pytorch's transform
# so I crop images according to example.py
#
# Crop images
from imageio import imread, imwrite
import os

# [(top, bottom), (left, right)]
crop_range = [(115, 734), (81, 874)]

def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
           crop_range[1][0]:crop_range[1][1], ...]

if not os.path.exists('crop_images'):
    os.mkdir('crop_images')
for file in os.listdir('example_images'):
    if os.path.isfile(os.path.join('example_images', file)):
        # prepare images
        # read
        image = imread(os.path.join('example_images', file))
        # crop
        image = imcrop(image, crop_range)
        # save
        imwrite(os.path.join('crop_images', file), image)

#!/usr/bin/env python

from distutils.core import setup, Extension

setup(
    name='nnio',
    version='0.1.0',
    description='Simple neural network inference on devices',
    author='FastSense',
    author_email='',
    url='',
    packages=['nnio', 'nnio.zoo', 'nnio.zoo.edgetpu', 'nnio.zoo.openvino', 'nnio.zoo.onnx'],
    install_requires=[
        'opencv-python'
    ]
)

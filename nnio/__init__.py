__version__ = '0.3.0'

import nnio.zoo

# Base model class
from .model import Model

# Models for specific backends
from .edgetpu import EdgeTPUModel
from .openvino import OpenVINOModel
from .onnx import ONNXModel

# Preprocessing class
from .preprocessing import Preprocessing

# Output classes
from .output import DetectionBox

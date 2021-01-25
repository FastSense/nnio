# nnio
## Installation:

```
pip install nnio
```

## Low-level API
To write your own model, derive from `nnio.Model` class:

```python
import nnio

class MyModel(nnio.Model):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        # Do something with image
        # For example, classification
        return 'cat'

    def get_preprocessing(self):
        return nnio.Preprocessing(
            resize=(224, 224),
            dtype='float',
            divide_by_255=True,
            batch_dimension=True,
            channels_first=True,
        )
```

Then use this model as:

```python
# Create model
model = MyModel()
# Get preprocessing function
preproc = model.get_preprocessing()
# Load image from file and preprocess it
image = preproc('/path/to/image.png')
# Pass image to model
result = model(image)
```

## High-level API
This package is designed to use with EdgeTPU and OpenVINO devices.  

This is example of using model with EdgeTPU:

```python
# Create model
model = nnio.EdgeTPUModel(
    # Model path can be URL:
    model_path='https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_v2_1.0_224_quant.tflite',
    # Use CPU for now:
    device=None,
)
# Create preprocessor
preproc = nnio.Preprocessing(
    resize=(224, 224),
    dtype='uint8',
    padding=True,
    batch_dimension=True,
)

# Read input file
image = preproc('/path/to/image.png')

# Make prediction
class_scores = model(image)
```

## Ready to use models
Example using SSDMobileNetV1 for object detection:

```python
# Load model
model = nnio.zoo.onnx.detection.SSDMobileNetV1()

# Pass to the neural network
image_prepared = preproc(image_rgb)
boxes = model(image_prepared)
```

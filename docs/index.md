# nnio
nnio is a light-weight python package for easily running neural networks.  

It supports running models on CPU as well as some of the edge devices:

* [Google Coral Edge TPU](https://coral.ai/)
* Intel integrated GPUs
* [Intel Myriad VPU](https://www.intel.ru/content/www/ru/ru/products/processors/movidius-vpu/movidius-myriad-x.html)

For each device there is a specific API and model saving format. We wrap all those in a single class-based API.

There are 3 ways one can use nnio:

1. Loading your saved models for inference
2. Using already prepared models from our model zoo: `nnio.zoo`
3. Using our API to wrap around your own custom models.

### Option 1: Loading your saved models for inference

nnio provides three classes for loading models in different formats:

* `nnio.ONNXModel`
* `nnio.EdgeTPUModel`
* `nnio.OpenVINOModel`

Loaded models can be simply called as functions. Look at the example:

```python
# Create model and put it on TPU device
model = nnio.EdgeTPUModel(
    model_path='path/to/model_quant_edgetpu.tflite',
    device='TPU:0',
)
# Create preprocessor
preproc = nnio.Preprocessing(
    resize=(224, 224),
    dtype='uint8',
    padding=True,
    batch_dimension=True,
)

# Preprocess your numpy image
image = preproc(image_rgb)

# Make prediction
class_scores = model(image)
```

`nnio.Preprocessing` class is described [here](./preprocessing.md).

### Option 2: Using already prepared models from our model zoo

Some popular models are already built in nnio. Example of using SSD MobileNet object detection model on CPU:

```python
# Load model
model = nnio.zoo.onnx.detection.SSDMobileNetV1()

# Get preprocessing function
preproc = model.get_preprocessing()

# Preprocess your numpy image
image = preproc(image_rgb)

# Make prediction
boxes = model(image)
```

More ready-to-use models are described [here](./model_zoo.md)

Here `boxes` is a list of `nnio.DetectionBox` instances. This class in described [here](./output.md).

### Option 3: Using our API to wrap around your own custom models
`nnio.Model` is an abstract class from which all models in nnio are derived. It is easy to use by redefining `forward` method:

```python
class MyClassifier(nnio.Model):
    def __init__(self):
        super().__init__()
        self.model = SomeModel()

    def forward(self, image):
        # Do something with image
        result = self.model(image)
        # For example, classification
        if result == 0:
            return 'person'
        else:
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

We also recommend to define `get_preprocessing` method like in `nnio.zoo` models.  
We encourage users to wrap their loaded models in such classes. Examples of this can be found [here](./model_class.md).

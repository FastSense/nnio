.. _nnio.zoo:

Model Zoo
===================

Using pretrained models
-----------------------

Some popular models are already built in nnio. Example of using SSD MobileNet object detection model on CPU:

.. code-block:: python

    # Load model
    model = nnio.zoo.onnx.detection.SSDMobileNetV1()

    # Get preprocessing function
    preproc = model.get_preprocessing()

    # Preprocess your numpy image
    image = preproc(image_rgb)

    # Make prediction
    boxes = model(image)

Here :code:`boxes` is a list of :ref:`nnio.DetectionBox` instances.


.. _basic_usage:

Basic Usage
===================

Using your saved models
-----------------------

nnio provides three classes for loading models in different formats:

* :ref:`nnio.ONNXModel`
* :ref:`nnio.EdgeTPUModel`
* :ref:`nnio.OpenVINOModel`

Loaded models can be simply called as functions on numpy arrays. Look at the example:

.. code-block:: python

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

See also :ref:`nnio.Preprocessing` documentation.

Below is the description of the basic nnio model classes:


.. _nnio.ONNXModel:

nnio.ONNXModel
--------------

.. autoclass:: nnio.ONNXModel
    :members:
    :special-members:


.. _nnio.EdgeTPUModel:

nnio.EdgeTPUModel
------------------

.. autoclass:: nnio.EdgeTPUModel
    :members:
    :special-members:


.. _nnio.OpenVINOModel:

nnio.OpenVINOModel
------------------

.. autoclass:: nnio.OpenVINOModel
    :members:
    :special-members:

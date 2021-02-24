What is it
==================================================================================
**nnio** is a light-weight python package for easily running neural networks.

It supports running models on CPU as well as some of the edge devices:

* `Google Coral Edge TPU <https://coral.ai/>`_
* Intel integrated GPUs
* `Intel Myriad VPU <https://www.intel.ru/content/www/ru/ru/products/processors/movidius-vpu/movidius-myriad-x.html>`_

For each device there exists an own library and a model format. We wrap all those in a single well-defined python package.

How to intall
-------------
nnio is simply installed with pip, but it requires some additional libraries.
See :ref:`installation`.


How to use it
-------------

There are 3 ways one can use nnio:

1. Loading your saved models for inference - :ref:`basic_usage`
2. Using already prepared models from our model zoo: :ref:`nnio.zoo`
3. Using our API to wrap around your own custom models. :ref:`extending`




.. toctree::
    :maxdepth: 2

    install
    basic_usage
    zoo
    utils
    extending
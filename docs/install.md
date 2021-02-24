# Istallation

Basic installation is simple:

```
pip install nnio
```

To use one of three backends, additional installs are needed:

## ONNX
To work with onnx backend, install onnxruntime package:

```
pip install onnxruntime
```

## EdgeTPU
To work with EdgeTPU models, `tflite_runtime` is required.  
See [installation guide](https://www.tensorflow.org/lite/guide/python).

If you intend to only use CPU inference, tensorflow installation will be enough.

## OpenVINO
To work with OpenVINO models user needs to install openvino package.  
The easiest way to do it is to use `openvino/ubuntu18_runtime` docker.  
The following command allows to pass all Myriad and GPU devices into docker container:

```
docker run -itu root:root --rm \
-v /var/tmp:/var/tmp \
--device /dev/dri:/dev/dri --device-cgroup-rule='c 189:* rmw' \
-v /dev/bus/usb:/dev/bus/usb \
-v /etc/timezone:/etc/timezone:ro \
-v /etc/localtime:/etc/localtime:ro \
-v "$(pwd):/input" openvino/ubuntu18_runtime
```
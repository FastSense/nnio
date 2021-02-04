from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...edgetpu import EdgeTPUModel
from ...output import DetectionBox


class SSDMobileNet(Model):
    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_{}_coco_quant_postprocess.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_{}_coco_quant_postprocess_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/coco_labels.txt'

    def __init__(
        self,
        device='CPU',
        version='v2',
        threshold=0.5
    ):
        '''
        input:
        - device: str
            "CPU" by default.
            Set "TPU" or "TPU:0" to use the first EdgeTPU device.
            Set "TPU:1" to use the second EdgeTPU device etc.
        - version: str
            Either "v1" or "v2"
        - threshold: float
            Detection threshold
        '''
        super().__init__()

        self.threshold = threshold

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU.format(version)
        else:
            model_path = self.URL_TPU.format(version)
        self.model = EdgeTPUModel(model_path, device)

        # Load labels from text file
        labels_path = utils.file_from_url(self.URL_LABELS, 'labels_google')
        self.labels = {
            int(line.split()[0]): line.strip().split()[1]
            for line in open(labels_path)
        }

    def forward(self, image, return_info=False):
        out = self.model(image, return_info=return_info)
        if return_info:
            (boxes, classes, scores, _num_detections), info = out
        else:
            boxes, classes, scores, _num_detections = out
        # Parse output
        out_boxes = []
        for i in range(len(boxes[0])):
            if scores[0, i] < self.threshold:
                continue
            x_1, y_1, x_2, y_2 = boxes[0, i]
            label = self.labels[int(classes[0, i])]
            score = scores[0, i]
            out_boxes.append(
                DetectionBox(x_1, y_1, x_2, y_2, label, score)
            )
        if return_info:
            return out_boxes, info
        else:
            return out_boxes

    def get_preprocessing(self):
        return Preprocessing(
            resize=(300, 300),
            dtype='uint8',
            batch_dimension=True,
        )


class SSDMobileNetFace(Model):
    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_face_quant_postprocess.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'

    def __init__(
        self,
        device='CPU',
        threshold=0.5
    ):
        '''
        input:
        - device: str
            "CPU" by default.
            Set "TPU" or "TPU:0" to use the first EdgeTPU device.
            Set "TPU:1" to use the second EdgeTPU device etc.
        - threshold: float
            Detection threshold. It affects sensitivity of the detector.
        '''
        super().__init__()

        self.threshold = threshold

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU
        else:
            model_path = self.URL_TPU
        self.model = EdgeTPUModel(model_path, device)

    def forward(self, image, return_info=False):
        out = self.model(image, return_info=return_info)
        if return_info:
            (boxes, _, scores, _num_detections), info = out
        else:
            boxes, _, scores, _num_detections = out
        # Parse output
        out_boxes = []
        for i in range(len(boxes[0])):
            if scores[0, i] < self.threshold:
                continue
            x_1, y_1, x_2, y_2 = boxes[0, i]
            label = 'face'
            score = scores[0, i]
            out_boxes.append(
                DetectionBox(x_1, y_1, x_2, y_2, label, score)
            )
        if return_info:
            return out_boxes, info
        else:
            return out_boxes

    def get_preprocessing(self):
        return Preprocessing(
            resize=(320, 320),
            dtype='uint8',
            batch_dimension=True,
        )

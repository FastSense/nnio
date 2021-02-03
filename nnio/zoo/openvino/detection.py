from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...openvino import OpenVINOModel
from ...output import DetectionBox

class SSDMobileNetV2(Model):
    URL_MODEL_BIN = 'https://github.com/FastSense/nnio/raw/development/models/openvino/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_fp16.bin'
    URL_MODEL_XML = 'https://github.com/FastSense/nnio/raw/development/models/openvino/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco_fp16.xml'
    URL_LABELS = 'https://github.com/amikelive/coco-labels/raw/master/coco-labels-paper.txt'

    def __init__(
        self,
        device='CPU',
        lite=True,
        threshold=0.5
    ):
        '''
        input:
        - device: str
            Choose Intel device:
            "CPU", "GPU", "MYRIAD"
        - threshold: float
            Detection threshold. It affects sensitivity of the detector.
        - lite: bool
            If True, use SSDLite version
        '''
        super().__init__()

        self.threshold = threshold

        path_bin = self.URL_MODEL_BIN
        path_xml = self.URL_MODEL_XML
        if lite:
            path_bin = path_bin.replace('ssd', 'ssdlite')
            path_xml = path_xml.replace('ssd', 'ssdlite')

        # Load model
        self.model = OpenVINOModel(path_bin, path_xml, device)

        # Load labels from text file
        labels_path = utils.file_from_url(self.URL_LABELS, 'labels')
        self.labels = [
            line.strip()
            for line in open(labels_path)
        ]

    def forward(self, image, return_info=False):
        results = self.model(image, return_info=return_info)
        if return_info:
            results, info = results
        # Parse output
        out_boxes = []
        for res in results[0, 0]:
            _, label, score, y_1, x_1, y_2, x_2 = res
            if score < self.threshold:
                continue
            label = self.labels[int(label) - 1]
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
            dtype='float32',
            channels_first=True,
            batch_dimension=True,
            bgr=True,
        )

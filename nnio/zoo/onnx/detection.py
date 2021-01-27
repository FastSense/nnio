from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...onnx import ONNXModel
from ...output import DetectionBox

class SSDMobileNetV1(Model):
    URL_MODEL = 'https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx'
    URL_LABELS = 'https://github.com/amikelive/coco-labels/raw/master/coco-labels-paper.txt'

    def __init__(self):
        '''
        '''
        super().__init__()

        # Load model
        self.model = ONNXModel(self.URL_MODEL)

        # Load labels from text file
        labels_path = utils.file_from_url(self.URL_LABELS, 'labels')
        self.labels = [
            line.strip()
            for line in open(labels_path)
        ]

    def forward(self, image):
        '''
        input:
        - image: np array
            Batch of input images
        output:
        - out_batch: list
            List of lists of boxes. First index is image id inside batch. Second index is box's number
        '''
        boxes, classes, scores, num_detections = self.model(image)
        # Parse output
        classes = classes - 1
        out_batch = []
        for batch_i in range(len(boxes)):
            out_boxes = []
            for i in range(int(num_detections[batch_i])):
                x_1, y_1, x_2, y_2 = boxes[batch_i, i]
                label = self.labels[int(classes[batch_i, i])]
                score = scores[batch_i, i]
                out_boxes.append(
                    DetectionBox(x_1, y_1, x_2, y_2, label, score)
                )
            out_batch.append(out_boxes)
        return out_batch

    def get_preprocessing(self):
        return Preprocessing(
            resize=(224, 224),
            dtype='uint8',
            batch_dimension=True,
        )

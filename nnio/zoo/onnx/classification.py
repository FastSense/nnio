import numpy as np

from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...onnx import ONNXModel


class MobileNetV2(Model):
    URL_MODEL = 'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx'
    URL_LABELS = 'https://github.com/onnx/models/raw/master/vision/classification/synset.txt'

    def __init__(self):
        '''
        '''
        super().__init__()

        # Load model
        self.model = ONNXModel(self.URL_MODEL)

        # Load labels from text file
        labels_path = utils.file_from_url(self.URL_LABELS, 'labels')
        self.labels = [
            ' '.join(line.strip().split()[1:])
            for line in open(labels_path)
        ]

    def forward(self, image, return_scores=False):
        scores = self.model(image)[0][0]
        label = self.labels[scores.argmax()]
        if return_scores:
            return label, scores
        else:
            return label

    def get_preprocessing(self):
        return Preprocessing(
            resize=(224, 224),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            scales=1 / np.array([0.229, 0.224, 0.225]),
            padding=False,
            batch_dimension=True,
            channels_first=True,
        )

from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import onnx as _onnx


class MobileNetV2(_model.Model):
    '''
    MobileNetV2 classifier trained on ImageNet

    Model is taken from the `ONNX Model Zoo <https://github.com/onnx/models>`_.
    '''

    URL_MODEL = 'https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx'
    URL_LABELS = 'https://github.com/onnx/models/raw/master/vision/classification/synset.txt'

    def __init__(self):
        '''
        '''
        super().__init__()

        # Load model
        self.model = _onnx.ONNXModel(self.URL_MODEL)

        # Load labels from text file
        labels_path = _utils.file_from_url(self.URL_LABELS, 'labels')
        self._labels = [
            ' '.join(line.strip().split()[1:])
            for line in open(labels_path)
        ]

    def forward(self, image, return_scores=False):
        '''
        :parameter image: np array.
            Input image
        :parameter return_scores: bool.
            If ``True``, return class scores.
        :return: ``str``: class label.
        '''
        scores = self.model(image)[0]
        label = self.labels[scores.argmax()]
        if return_scores:
            return label, scores
        else:
            return label

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(224, 224),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            padding=False,
            batch_dimension=True,
            channels_first=True,
        )

    @property
    def labels(self):
        '''
        :return: list of ImageNet classification labels
        '''
        return self._labels

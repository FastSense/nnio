from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...edgetpu import EdgeTPUModel


class MobileNet(Model):
    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_{}_1.0_224_quant.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/mobilenet_{}_1.0_224_quant_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/imagenet_labels.txt'

    def __init__(self, device=None, version='v2'):
        '''
        input:
        - device: str or None
            Set ":0" to use the first EdgeTPU device.
            Set ":1" to use the second EdgeTPU device.
            Same for other devices if they are present.
            Leave None to use CPU
        - version: str
            Either "v1" or "v2"
        '''
        super().__init__()

        # Load model
        if device is None:
            model_path = self.URL_CPU.format(version)
        else:
            model_path = self.URL_TPU.format(version)
        self.model = EdgeTPUModel(model_path, device)

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
            dtype='uint8',
            padding=True,
            batch_dimension=True,
        )

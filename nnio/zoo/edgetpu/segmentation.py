from ... import utils
from ...preprocessing import Preprocessing

from ...model import Model
from ...edgetpu import EdgeTPUModel


class DeepLabV3(Model):
    URL_CPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant.tflite'
    URL_TPU = 'https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite'
    URL_LABELS = 'https://github.com/google-coral/edgetpu/raw/master/test_data/pascal_voc_segmentation_labels.txt'

    def __init__(self, device='CPU', version='v2'):
        '''
        input:
        - device: str
            "CPU" by default.
            Set "TPU" or "TPU:0" to use the first EdgeTPU device.
            Set "TPU:1" to use the second EdgeTPU device etc.
        - version: str
            Either "v1" or "v2"
        '''
        super().__init__()

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU.format(version)
        else:
            model_path = self.URL_TPU.format(version)
        self.model = EdgeTPUModel(model_path, device)

        # Load labels from text file
        labels_path = utils.file_from_url(self.URL_LABELS, 'labels')
        self.labels = []
        for line in open(labels_path):
            if line.strip() != '':
                self.labels.append(line.strip())
            else:
                break

    def forward(self, image):
        '''
        input:
        - image: numpy array
            Image, prepared using preprocessing function returned by get_preprocessing()

        output: numpy array
            Segmentation map of the same size as the input image: shape=[batch, 513, 513]
            For each pixel gives an integer denoting class.
            Class labels are available through .labels attribute of this object.
        '''
        segmentation = self.model(image)[0][0]
        return segmentation

    def get_preprocessing(self):
        return Preprocessing(
            resize=(513, 513),
            dtype='uint8',
            padding=False,
            batch_dimension=True,
        )

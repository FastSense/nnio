from ... import utils as _utils
from ... import preprocessing as _preprocessing

from ... import model as _model
from ... import edgetpu as _edgetpu


class OSNet(_model.Model):
    URL_CPU = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_quant.tflite'
    URL_TPU = 'https://github.com/FastSense/nnio/raw/master/models/person-reid/osnet_x1_0/osnet_x1_0_quant_edgetpu.tflite'

    def __init__(
        self,
        device='CPU',
    ):
        '''
        input:
        - device: str
            "CPU" by default.
            Set "TPU" or "TPU:0" to use the first EdgeTPU device.
            Set "TPU:1" to use the second EdgeTPU device etc.
        '''
        super().__init__()

        # Load model
        if device == 'CPU':
            model_path = self.URL_CPU
        else:
            model_path = self.URL_TPU
        self.model = _edgetpu.EdgeTPUModel(model_path, device)

    def forward(self, image, return_info=False):
        out = self.model(image, return_info=return_info)
        if return_info:
            vector, info = out
        else:
            vector = out
        vector = vector[0]
        if return_info:
            return vector, info
        else:
            return vector

    def get_preprocessing(self):
        return _preprocessing.Preprocessing(
            resize=(128, 256),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            channels_first=False,
            batch_dimension=True,
            padding=True,
        )

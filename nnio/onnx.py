import time

from .model import Model
from . import utils

class ONNXModel(Model):
    def __init__(
        self,
        model_path,
    ):
        '''
        input:
        - model_path: str
            url or path to the .onnx model
        '''
        super().__init__()
        # Download file from internet
        if utils.is_url(model_path):
            model_path = utils.file_from_url(model_path, 'models')
        # Load model and create inference session
        import onnxruntime as rt
        self.sess = rt.InferenceSession(model_path)

    def forward(self, *inputs, return_time=False):
        '''
        input:
        - *inputs: list of arguments
            Input numpy arrays
        - return_time: bool
            If True, will return inference time
        '''
        assert len(inputs) == len(self.get_input_details())
        # List output names
        outputs = [
            info['name']
            for info in self.get_output_details()
        ]
        # Convert input to a dict
        inputs = {
            info['name']: inp
            for (info, inp) in zip(self.get_input_details(), inputs)
        }
        # Run network and measure time
        start = time.time()
        results = self.sess.run(outputs, inputs)
        end = time.time()
        times = {
            'invoke': end - start,
        }
        if return_time:
            return results, times
        else:
            return results

    def get_input_details(self):
        return [
            {
                'name': info.name,
                'shape': info.shape,
                'dtype': info.type,
            }
            for info in self.sess.get_inputs()
        ]

    def get_output_details(self):
        return [
            {
                'name': info.name,
                'shape': info.shape,
                'dtype': info.type,
            }
            for info in self.sess.get_outputs()
        ]

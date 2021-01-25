import platform
import time

from .model import Model
from . import utils

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

class EdgeTPUModel(Model):
    def __init__(
        self,
        model_path,
        device=None
    ):
        '''
        input:
        - model_path: str
            url or path to the tflite model
        - device: str or None
            Set ":0" to use the first EdgeTPU device.
            Set ":1" to use the second EdgeTPU device.
            Same for other devices if they are present.
            Leave None to use CPU
        '''
        super().__init__()
        # Download file from internet
        if utils.is_url(model_path):
            model_path = utils.file_from_url(model_path, 'models')
        # Create interpreter
        self.interpreter = self.make_interpreter(model_path, device)
        self.interpreter.allocate_tensors()

    def forward(self, *inputs, return_time=False):
        '''
        input:
        - *inputs: list of arguments
            Input numpy arrays
        - return_time: bool
            If True, will return inference time
        '''
        assert len(inputs) == self.n_inputs
        start = time.time()
        # Put input tensors into model
        for i in range(self.n_inputs):
            tensor = self.input_tensor(i)
            tensor[:, :, :, :] = inputs[i]
            del tensor
        before_invoke = time.time()
        # Call model
        self.interpreter.invoke()
        after_invoke = time.time()
        # Get results from the model
        results = [self.output_tensor(i) for i in range(self.n_outputs)]
        end = time.time()
        times = {
            'assign': before_invoke - start,
            'invoke': after_invoke - before_invoke,
            'getres': end - after_invoke,
        }
        if return_time:
            return results, times
        else:
            return results

    def get_input_details(self):
        return [
            {
                'name': inp['name'],
                'shape': inp['shape'],
                'dtype': str(inp['dtype']),
            }
            for inp in self.interpreter.get_input_details()
        ]

    def get_output_details(self):
        return [
            {
                'name': inp['name'],
                'shape': inp['shape'],
                'dtype': str(inp['dtype']),
            }
            for inp in self.interpreter.get_output_details()
        ]

    @staticmethod
    def make_interpreter(model_file, device=None):
        import tflite_runtime.interpreter as tflite
        if device is not None:
            return tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[
                    tflite.load_delegate(
                        EDGETPU_SHARED_LIB,
                        {'device': device})
                ])
        else:
            return tflite.Interpreter(
                model_path=model_file)

    def input_tensor(self, i=0):
        '''
        Returns input tensor view as function returning numpy array
        '''
        tensor_index = self.interpreter.get_input_details()[i]['index']
        return self.interpreter.tensor(tensor_index)()

    def output_tensor(self, i=0):
        """Returns output tensor view."""
        tensor_index = self.interpreter.get_output_details()[i]['index']
        tensor = self.interpreter.get_tensor(tensor_index)
        return tensor

    @property
    def n_inputs(self):
        ''' number of input tensors '''
        return len(self.interpreter.get_input_details())

    @property
    def n_outputs(self):
        ''' number of output tensors '''
        return len(self.interpreter.get_output_details())


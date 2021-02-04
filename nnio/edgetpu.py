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
        device='CPU'
    ):
        '''
        input:
        - model_path: str
            url or path to the tflite model
        - device: str
            "CPU" by default.
            Set "TPU" or "TPU:0" to use the first EdgeTPU device.
            Set "TPU:1" to use the second EdgeTPU device etc.
        '''
        super().__init__()
        # Download file from internet
        if utils.is_url(model_path):
            model_path = utils.file_from_url(model_path, 'models')
        # Create interpreter
        assert device == 'CPU' or device.split(':')[0] == 'TPU' or device[0] == ':'
        self.interpreter = self.make_interpreter(model_path, device)
        self.interpreter.allocate_tensors()

    def forward(self, *inputs, return_info=False):
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
        # Return results
        if return_info:
            info = {
                'assign_time': before_invoke - start,
                'invoke_time': after_invoke - before_invoke,
            }
            return results, info
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
    def make_interpreter(model_file, device='CPU'):
        ' Load model and create tflite interpreter '
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            print('''
            Warning: tflite_runtime is not installed.
            
            Please follow these instructions to install driver:
            
            https://coral.ai/docs/m2/get-started/#2a-on-linux
            
            And these instructions to install tflite_runtime:
            
            https://www.tensorflow.org/lite/guide/python
            ''')
            if device == 'CPU':
                print('Trying to use tensorflow version on CPU')
                import tensorflow.lite as tflite
            else:
                raise ImportError
        if device != 'CPU':
            if device == 'TPU':
                device = 'TPU:0'
            return tflite.Interpreter(
                model_path=model_file,
                experimental_delegates=[
                    tflite.load_delegate(
                        EDGETPU_SHARED_LIB,
                        {
                            'device': device.replace('TPU', '')
                        }
                    )
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


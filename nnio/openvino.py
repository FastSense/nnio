import time

from .model import Model
from . import utils

class OpenVINOModel(Model):
    def __init__(
        self,
        model_bin,
        model_xml,
        device='CPU',
    ):
        '''
        input:
        - model_bin: str
            url or path to the openvino binary model file
        - model_xml: str
            url or path to the openvino xml model file
        - device: str
            Choose Intel device:
            "CPU", "GPU", "MYRIAD"
        '''
        super().__init__()

        # Download files from internet
        if utils.is_url(model_bin):
            model_bin = utils.file_from_url(model_bin, 'models')
        if utils.is_url(model_xml):
            model_xml = utils.file_from_url(model_xml, 'models')

        # Create interpreter
        self.device = device
        self.ie, self.net = self.make_interpreter(model_xml, model_bin, device)

    def forward(self, inputs, return_info=False):
        '''
        input:
        - inputs: numpy array
            Input data
        - return_time: bool
            If True, will return inference time
        '''
        # Find name of the input to the model
        input_name = list(self.net.input_info.keys())[0]
        # Call model
        start = time.time()
        out = self.net.infer({input_name: inputs})
        end = time.time()
        # Process output a little
        if len(out.keys()) == 1:
            out = out[list(out.keys())[0]]
        if return_info:
            info = {
                'invoke_time': end - start,
            }
            if self.device == 'MYRIAD':
                info['temperature'] = self.ie.get_metric(metric_name="DEVICE_THERMAL", device_name="MYRIAD")
            return out, info
        else:
            return out

    @staticmethod
    def make_interpreter(model_xml, model_bin, device):
        'Load model and create openvino interpreter'
        try:
            from openvino.inference_engine import IECore
        except ImportError:
            print('''
            Warning: openvino is not installed.
            
            Please install openvino or use openvino docker container.
            ''')
            raise ImportError
        ie = IECore()
        net = ie.read_network(model_xml, model_bin)
        net = ie.load_network(net, device)
        return ie, net


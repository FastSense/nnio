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
            If there are multiple devices in your system, use indeces:
            "MYRIAD:0"
        '''
        super().__init__()

        # Download files from internet
        if utils.is_url(model_bin):
            model_bin = utils.file_from_url(model_bin, 'models')
        if utils.is_url(model_xml):
            model_xml = utils.file_from_url(model_xml, 'models')

        # Create interpreter
        self.ie, self.net, self.device = self.make_interpreter(model_xml, model_bin, device)

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
        # Measure temperature
        if self.device.startswith('MYRIAD'):
            temperature = self.ie.get_metric(metric_name="DEVICE_THERMAL", device_name=self.device)
            if utils.LOG_TEMPERATURE:
                utils.log_temperature(self.device, temperature)
        # Return results
        if return_info:
            info = {
                'invoke_time': end - start,
            }
            if self.device.startswith('MYRIAD'):
                info['temperature'] = temperature
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
        # Get list of MYRIAD devices
        if 'MYRIAD' in device:
            myriads = [
                dev for dev in ie.available_devices
                if dev.startswith('MYRIAD')
            ]
            if ':' in device:
                device, idx = device.split(':')
                idx = int(idx)
            else:
                idx = 0
            device = myriads[idx]
        # Load model on device
        net = ie.read_network(model_xml, model_bin)
        print('Loading model to:', device)
        net = ie.load_network(net, device)
        return ie, net, device

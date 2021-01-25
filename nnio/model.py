import abc


class Model(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        '''
        This function is called when the model is called.
        Override this function with your model's behavior
        '''

    def get_preprocessing(self):
        '''
        Overriding this function is recommended.
        It should return the callable object for image preprocessing
        For example:
        return nnio.Preprocessing(
            resize=(224, 224),
            dtype='uint8',
            padding=True,
            batch_dimension=True,
        )
        '''

    def get_input_details(self):
        '''
        Overriding this function is recommended.
        It should return human-readable input details.
        For example:
        [
            {
                'name': 'input_image'
                'shape': [1, 224, 224, 3]
                'dtype': 'uint8'
            }
        ]
        '''

    def get_output_details(self):
        '''
        Overriding this function is recommended.
        It should return human-readable output details.
        For example:
        [
            {
                'name': 'detection_boxes'
                'shape': [10, 14, 14, 4]
                'type': 'float'
            }
            {
                'name': 'detection_classes'
                'shape': [10]
                'type': 'float'
            }
        ]
        '''

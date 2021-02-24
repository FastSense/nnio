import abc


class Model(abc.ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        This function is called when the model is called.
        Override this function with your model's behavior
        """

    def get_preprocessing(self):
        """
        Returns nnio.Preprocessing object.
        """

    def get_input_details(self):
        """
        Returns human-readable model input details.
        """

    def get_output_details(self):
        """
        Returns human-readable model output details.
        """

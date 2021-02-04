import cv2
import numpy as np
import os

from .model import Model
from . import utils

class Preprocessing(Model):
    def __init__(
        self,
        resize=None,
        dtype='uint8',
        divide_by_255=False,
        means=None,
        scales=None,
        padding=False,
        channels_first=False,
        batch_dimension=False,
        bgr=False,
    ):
        '''
        input:
        - resize: tuple or None
            (width, height) - the new size of image
        - dtype: str on np.dtype
            Data type
        - means: float of list or None
            Substract these values from each channel
        - scales: float of list or None
            Multipy each channel by these values
        - padding: bool
            If True, images will be resized with the same aspect ratio
        - channels_first: bool
            If True, image will be returned in [B]CHW format.
            If False, [B]HWC
        - batch_dimension: bool
            If True, add first dimension of size 1
        - bgr: bool
            If True, change channels to BRG order
            If False, keep the RGB order
        '''
        self.resize = resize
        self.dtype = dtype
        self.divide_by_255 = divide_by_255
        if divide_by_255:
            MSG = 'If dividing image by 255, specify float data type'
            assert 'float' in dtype, MSG
        self.means = means
        self.scales = scales
        self.padding = padding
        self.channels_first = channels_first
        self.batch_dimension = batch_dimension
        self.bgr = bgr

    def forward(self, image, return_shape=False):
        '''
        Preprocess the image.
        input:
        - image: np.ndarray of type uint8 or str
            RGB image
            If str, it will be concerned as image path.
        - return_shape: bool
            If True, will return shape of the original image
        '''
        # Read image
        if isinstance(image, str):
            image = self.read_image(image)
        orig_shape = image.shape
        assert str(image.dtype) == 'uint8'

        # Convert colors
        if self.bgr:
            image = image[:, :, ::-1]

        # Resize image
        if self.resize is not None:
            image = self.resize_image(image, self.resize, self.padding)

        # Divide by 255
        if self.divide_by_255:
            image = image / 255

        # Shift and scale
        if self.means is not None:
            image = image - np.array(self.means)[None, None]
        if self.scales is not None:
            image = image * np.array(self.scales)[None, None]

        # Change shape
        if self.channels_first:
            image = image.transpose([2, 0, 1])
        if self.batch_dimension:
            image = image[None]

        # Change datatype
        image = image.astype(self.dtype)

        if return_shape:
            return image.copy(), orig_shape
        else:
            return image.copy()

    @staticmethod
    def read_image(path):
        ''' Read image from file or url '''
        # Download image if path is url
        is_url = utils.is_url(path)
        if is_url:
            path = utils.file_from_url(path, 'temp')
        # Read image
        # pylint: disable=no-member
        image = cv2.imread(path)
        # Delete temporary file
        if is_url:
            os.remove(path)
        # Throw exception
        if image is None:
            raise BaseException('Cannot read ' + path)
        # Convert from BGR to RGB
        image = image[:, :, ::-1]
        return image

    @staticmethod
    def resize_image(image, resize, padding=False):
        ''' Resize image
        input:
        - image
        - resize: tuple
            (width, height) - the new size
        - padding: bool
            If True, image will be resized with the same aspect ratio
        '''
        if not padding:
            # pylint: disable=no-member
            image = cv2.resize(image, resize)
        else:
            # Resize saving the aspect ratio
            ratio_0 = image.shape[1] / resize[0]
            ratio_1 = image.shape[0] / resize[1]
            ratio = max(ratio_0, ratio_1)
            new_size = (
                int(image.shape[1] / ratio),
                int(image.shape[0] / ratio)
            )
            # pylint: disable=no-member
            image = cv2.resize(image, new_size)
            # Pad with zeros
            padding = np.zeros([resize[1], resize[0], 3],
                               dtype=image.dtype)
            start_0 = (padding.shape[0] - image.shape[0]) // 2
            start_1 = (padding.shape[1] - image.shape[1]) // 2
            padding[
                start_0: start_0 + image.shape[0],
                start_1: start_1 + image.shape[1],
            ] = image
            image = padding.copy()
        return image

    def __str__(self):
        ''' Outputs preprocessing parameters as a string '''
        s = 'Preprocessing(resize={}, dtype={}, divide_by_255={}, means={}, scales={}, padding={}, channels_first={}, batch_dimension={}, bgr={})'
        s = s.format(
            self.resize,
            self.dtype,
            self.divide_by_255,
            self.means,
            self.scales,
            self.padding,
            self.channels_first,
            self.batch_dimension,
            self.bgr,
        )
        return s

import cv2
import numpy as np
import os

from . import model as _model
from . import utils as _utils

class Preprocessing(_model.Model):
    '''
    This class provides functionality of the image preprocessing.

    Example::

        preproc = nnio.Preprocessing(
            resize=(224, 224),
            dtype='float32',
            divide_by_255=True,
            means=[0.485, 0.456, 0.406],
            stds=[0.229, 0.224, 0.225],
            batch_dimension=True,
            channels_first=True,
        )

        # Use with numpy image
        image_preprocessed = preproc(image_rgb)

        # Or use to read image from disk
        image_preprocessed = preproc('path/to/image.png')

        # Or use to read image from the web
        image_preprocessed = preproc('http://www.example.com/image.png')

    Object of this type is returned every time you call ``get_preprocessing()`` method of any model from :ref:`nnio.zoo`.
    '''
    def __init__(
        self,
        resize=None,
        dtype='uint8',
        divide_by_255=False,
        means=None,
        stds=None,
        scales=None,
        padding=False,
        channels_first=False,
        batch_dimension=False,
        bgr=False,
    ):
        '''
        :parameter resize: ``None`` or ``tuple``.
            (width, height) - the new size of image
        :parameter dtype: ``str`` on ``np.dtype``.
            Data type
        :parameter means: ``float`` or ``list`` or ``None``.
            Substract these values from each channel
        :parameter stds: `float`` or ``list`` or ``None``.
            Divide each channel by these values
        :parameter scales: ``float`` or ``list`` or ``None``.
            Multipy each channel by these values
        :parameter padding: ``bool``.
            If ``True``, images will be resized with the same aspect ratio
        :parameter channels_first: ``bool``.
            If ``True``, image will be returned in ``[B]CHW`` format.
            If ``False``, ``[B]HWC``.
        :parameter batch_dimension: ``bool``.
            If ``True``, add first dimension of size 1.
        :parameter bgr: ``bool``.
            If ``True``, change channels to BRG order.
            If ``False``, keep the RGB order.
        '''
        self.resize = resize
        self.dtype = dtype
        self.divide_by_255 = divide_by_255
        if divide_by_255:
            MSG = 'If dividing image by 255, specify float data type'
            assert 'float' in dtype, MSG
        self.means = means
        if stds is not None and scales is not None:
            raise BaseException('Either scales or stds may be specified, not both')
        self.stds = stds
        self.scales = scales
        self.padding = padding
        self.channels_first = channels_first
        self.batch_dimension = batch_dimension
        self.bgr = bgr

    def forward(self, image, return_original=False):
        '''
        Preprocess the image.

        :parameter image: np.ndarray of type ``uint8`` or ``str``
            RGB image
            If ``str``, it will be concerned as image path.
        :parameter return_original: ``bool``.
            If ``True``, will return tuple of ``(preprocessed_image, original_image)``
        '''
        # Read image
        if isinstance(image, str):
            image = self._read_image(image)
        if return_original:
            orig_image = image.copy()
        if str(image.dtype) != 'uint8':
            raise BaseException('Input image data type for preprocessor must be uint8')

        # Convert colors
        if self.bgr:
            image = image[:, :, ::-1]

        # Resize image
        if self.resize is not None:
            image = self._resize_image(image, self.resize, self.padding)

        # Divide by 255
        if self.divide_by_255:
            image = image / 255

        # Shift and scale
        if self.means is not None:
            image = image - np.array(self.means)[None, None]
        if self.stds is not None:
            image = image / np.array(self.stds)[None, None]
        if self.scales is not None:
            image = image * np.array(self.scales)[None, None]

        # Change shape
        if self.channels_first:
            image = image.transpose([2, 0, 1])
        if self.batch_dimension:
            image = image[None]

        # Change datatype
        image = image.astype(self.dtype)

        if return_original:
            return image.copy(), orig_image
        else:
            return image.copy()

    @staticmethod
    def _read_image(path):
        ''' Read image from file or url '''
        # Download image if path is url
        is_url = _utils.is_url(path)
        if is_url:
            path = _utils.file_from_url(path, 'temp')
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
    def _resize_image(image, resize, padding=False):
        ''' Resize image
        
        :parameter image: np.ndarray of type ``uint8`` or ``str``
            RGB image
            If ``str``, it will be concerned as image path.
        :parameter resize: ``None`` or ``tuple``.
            (width, height) - the new size of image
        :parameter padding: ``bool``.
            If ``True``, images will be resized with the same aspect ratio
        :return: np.array. Resized image.
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
        '''
        :return: full description of the ``Preprocessing`` object
        '''
        s = 'nnio.Preprocessing(resize={}, dtype={}, divide_by_255={}, means={}, stds={}, scales={}, padding={}, channels_first={}, batch_dimension={}, bgr={})'
        s = s.format(
            self.resize,
            self.dtype,
            self.divide_by_255,
            self.means,
            self.stds,
            self.scales,
            self.padding,
            self.channels_first,
            self.batch_dimension,
            self.bgr,
        )
        return s

    def __eq__(self, other):
        '''Compare two ``Preprocessing`` objects. Returns ``True`` only if all preprocessing parameters are the same.'''
        return str(self) == str(other)

import pathlib
from distutils.core import setup, Extension

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "readme.md").read_text()

setup(
    name='nnio',
    version='0.1.4',
    description='Simple neural network inference on devices',
    long_description='''
    See more details at the project's homepage: https://github.com/FastSense/nnio
    ''',
    author='FastSense',
    author_email='ruslan.baynazarov@fastsense.tech',
    url='https://github.com/FastSense/nnio',
    packages=['nnio', 'nnio.zoo', 'nnio.zoo.edgetpu', 'nnio.zoo.openvino', 'nnio.zoo.onnx'],
    install_requires=[
        'opencv-python'
    ]
)

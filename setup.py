from distutils.core import setup, Extension
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")
setup(
    name='nnio',
    version=get_version('nnio/__init__.py'),
    description='Simple neural network inference on devices',
    long_description='''
    See more details at the project's homepage: https://github.com/FastSense/nnio
    ''',
    author='FastSense',
    author_email='ruslan.baynazarov@fastsense.tech',
    url='https://github.com/FastSense/nnio',
    packages=['nnio', 'nnio.zoo', 'nnio.zoo.edgetpu', 'nnio.zoo.openvino', 'nnio.zoo.onnx'],
    install_requires=[
        'numpy',
        'opencv-python',
    ]
)

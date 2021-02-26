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


LONG_DESCRIPTION = read('readme.md')

setup(
    name='nnio',
    version=get_version('nnio/__init__.py'),
    description='Neural network inference on accelerators simplified',
    long_description=LONG_DESCRIPTION,
    author='FastSense',
    author_email='ruslan.baynazarov@fastsense.tech',
    url='https://github.com/FastSense/nnio',
    packages=['nnio', 'nnio.zoo', 'nnio.zoo.edgetpu', 'nnio.zoo.openvino', 'nnio.zoo.onnx'],
    license="MIT",
    install_requires=[
        'numpy',
        'opencv-python',
    ]
)

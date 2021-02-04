import urllib.request
import os
import pathlib
import getpass
import datetime

from . import __version__

PACKAGE_NAME = 'nnio'

# Temperature logging flag
LOG_TEMPERATURE = False
temperature_files = {}

def is_url(s):
    '''
    Check if input string is url or not
    '''
    return s.startswith('http://') or s.startswith('https://')

def file_from_url(url, category='other'):
    '''
    Downloads file to "/home/$USER/.cache/nnio/" if it does not exist already.
    Returns path to the file
    '''
    # Get base path for file
    base_path = os.path.join(
        '/home',
        getpass.getuser(),
        '.cache',
        PACKAGE_NAME,
        __version__,
        category,
    )
    # Create path if not exists
    if not os.path.exists(base_path):
        pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    # Get file path
    file_name = url.split('/')[-1]
    file_path = os.path.join(
        base_path,
        file_name
    )
    # Download file from the url
    if not os.path.exists(file_path):
        print('Downloading file from: {}'.format(url))
        urllib.request.urlretrieve(url, file_path)
        print('Downloaded to: {}'.format(file_path))
    else:
        print('Using cached file: {}'.format(file_path))

    return file_path


# Flag setter
def enable_logging_temperature(enable=True):
    global LOG_TEMPERATURE
    LOG_TEMPERATURE = enable

def log_temperature(device, temperature):
    # Get path to file
    if device in temperature_files:
        file_path = temperature_files[device]
    else:
        # Get base path for file
        base_path = os.path.join(
            '/home',
            getpass.getuser(),
            '.telemetry',
        )
        # Create path if not exists
        if not os.path.exists(base_path):
            pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
        # Make file name
        dev_id = device.replace('MYRIAD', '')
        time_format = "vpu{}_%Y-%m-%d_%H-%M-%S".format(dev_id)
        file_name = datetime.datetime.now().strftime(time_format)
        file_path = os.path.join(base_path, file_name)
        # Remember file path
        temperature_files[device] = file_path
        # Write first line to this file
        with open(file_path, 'w') as f:
            f.write('time,vpu_temp\n')
            f.flush()

    val_time = datetime.datetime.now().isoformat(timespec='milliseconds')
    val_vpu_temp = int(temperature)
    with open(file_path, 'a') as f:
        f.write('{},{}\n'.format(val_time, val_vpu_temp))
        f.flush()

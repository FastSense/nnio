import urllib.request
import os
import pathlib
import getpass

PACKAGE_NAME = 'nnio'

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
    else:
        print('Using cached file: {}'.format(file_path))

    return file_path

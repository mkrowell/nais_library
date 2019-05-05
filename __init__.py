# ------------------------------------------------------------------------------
# EXCEPTIONS
# ------------------------------------------------------------------------------
class fileNotFound(Exception):
    def __init__(self, directory, filename, msg = None):
        if msg is None:
            msg = ('The file {0} was not found in the directory: {1}').format(
                  filename, directory)
        super(fileNotFound, self).__init__(msg)
        self.directory = directory
        self.filename = filename


# ------------------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------------------
from collections import namedtuple
import certifi
import ctypes
import datetime
from dateutil.parser import parse
import os
from os.path import abspath, basename, dirname, exists, join, splitext
import pycurl
import re
import zipfile


# ------------------------------------------------------------------------------
# WEB
# ------------------------------------------------------------------------------
def download_url(url, destination, fileExtension = '.xls'):
    '''
    Write url to temp file.
    :param url: URL to download data from
    :param destination: Path to download data to
    :param fileExtension: extension to use for temp file

    :type url: string
    :type destination: string
    :type fileExtension: string
    :return: Path to temp file containing URL data
    :rtype: string
    '''
    tempFile = join(destination, 'temp_data' + fileExtension)
    with open(tempFile, 'wb') as f:
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(pycurl.CAINFO, certifi.where())
        c.setopt(pycurl.HTTP_VERSION, pycurl.CURL_HTTP_VERSION_1_0)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()
    return tempFile


# ------------------------------------------------------------------------------
# USER
# ------------------------------------------------------------------------------
def message_box_OK_Cancel(title, msg):
    '''
    Open a message box with OK and Cancel button options.

    :param title: Title of the message box
    :param msg: Message to show in message box

    :type title: string
    :type msg: string

    :return: Message box
    :rtype: cytpes.windll.user32.MessageBox
    '''
    return ctypes.windll.user32.MessageBoxW(0, msg, title, 1)


# ------------------------------------------------------------------------------
# FILE SYSTEM
# ------------------------------------------------------------------------------
def create_folder(parent, name):
    '''
    Create a folder with given name as a subdirectory of parent, if the
    folder does not already exist, and return path to the folder.

    :param parent: Directory in which the folder will be created
    :param name: Name of the folder to be created

    :type parent: string
    :type name: string

    :return: Path to folder
    :rtype: string
    '''
    folder = abspath(join(parent, name))
    if not exists(folder):
        try:
            os.makedirs(folder)
        except IOError as err:
            folderExistsCode = 17
            if err.errno != folderExistsCode:
                raise err
    return folder

def find_file(directory, filename):
    '''
    Search for a filename in the directory and return the full path. The
    extension does not need to be included, but the first filepath that
    matches will be returned.

    :param directory: Directory within which to search for file
    :param filename: Name of file, with or without extension, to search for

    :type directory: string
    :type filename: string

    :return: Path to file
    :rtype: string
    :raises fileNotFound: Filename cannot be located within the directory
    '''
    rex = re.compile(filename.lower())
    for root, dirs, files in os.walk(directory):
        for f in files:
            result = rex.search(f.lower())
            if result:
                return join(root, f)
    raise fileNotFound(directory, filename)

def file_parts(filepath):
    '''
    Return the path components of the file.

    :param filepath: Path to the file
    :type filepath: string
    :return: Directory, name, extension, and date (if present in name) of file
    :rtype: named tuple
    '''
    directory = dirname(filepath)
    name, extension = splitext(basename(filepath))
    fileparts = namedtuple('FileParts', 'directory name extension')
    return fileparts(directory, name, extension)

def rename_file(filepath, rename):
    '''
    Rename the filepath while keeping original extension.

    :param filepath: Path or name of file to be renamed
    :param rename: New name without extension

    :type filepath: string
    :type rename: string

    :return: Path or name of file with new name
    :rtype: string
    '''
    parts = file_parts(filepath)
    return join(parts.directory, rename + parts.extension)


# ------------------------------------------------------------------------------
# ZIP FILES
# ------------------------------------------------------------------------------
def extract_zip(filepath, destination):
    '''
    Extract zip file to the destination and return list of filenames contained
    within the archive.

    :param filepath: Filepath to the zip file to be unzipped
    :param destination: Location for the archive files to be extracted to

    :type filepath: string
    :type destination: string

    :return: List of filenames that have been extracted to the destination
    :rtype: list of strings
    '''
    zfile = zipfile.ZipFile(filepath, 'r')
    zfile.printdir()
    zfile.extractall(destination)
    zfile.close()
    return zfile.namelist()

def extract_file(filepath, destination):
    '''
    Return the path to the first file extracted from a zip file.
    '''
    zf = extract_zip(filepath, destination)
    if len(zf) == 1:
        name = zf[0]
        return join(destination, name)
    raise UserWarning('There are multiple files in the zip file.')


# ------------------------------------------------------------------------------
# PACKAGE IMPORTS
# ------------------------------------------------------------------------------
from .database import *

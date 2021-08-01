#
## imports
#
import os

import re

from pathlib import Path

from bs4 import BeautifulSoup

import xml.etree.ElementTree as ET

import collections, itertools

import numpy as np

import torch
import torch.utils.data
import torchvision

from PIL import Image

from matplotlib import pyplot as plt

import cv2
# import utils
# import transforms as T

from zipfile import ZipFile
from shutil import copy
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


    
####################################################
####################################################
"""
                               Part IV
                               
            ( getting zip file from google drive )


uses:

from zipfile import ZipFile
from shutil import copy
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials



"""

def drive_retrieve(fileId):
    # get access
    auth.authenticate_user()
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    # file name
    fileName = fileId + '.zip'

    # request, download??
    downloaded = drive.CreateFile({'id': fileId})
    downloaded.GetContentFile(fileName)

    # feed into ZipFile library
    ds = ZipFile(fileName)

    # extract
    ds.extractall()
    
    # now that it's been extracted, remove the zip file
    os.remove( fileName )
    print( 'Extracted zip file ' + fileName + '\n' )
    
   
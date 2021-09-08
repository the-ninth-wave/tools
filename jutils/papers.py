#
## imports
#
import os
import sys
import xml.etree.ElementTree as ET
import re
import time

# misc 2
import itertools
import logging
import json
from collections import OrderedDict

# math
import random
import math
import numpy as np

# torch
import torch
import torch.utils.data

# images and plots
import skimage.io
from PIL import Image, ImageDraw
import cv2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

"""
"""


import torchvision
from torchvision.io import read_image


"""
"""

from engine import train_one_epoch, evaluate
import utils
import transforms as T

"""
"""
from torchvision.transforms import functional as F

from torchvision.transforms import ToPILImage as to_pil_image
"""
"""
####
DATASET_DIR = "drive/Othercomputers/Normandie/GitHub/home/datasets/percolation_papers/ds_percolation_papers/papers"

# for display
TINT_COLOR = (255,255,255)    # white
TRANSPARENCY = .5    # degree of transparency
OPACITY = int( 255 * TRANSPARENCY )
OUTLINE_COLOR= (128,128,128) #grey?
"""
"""
####









####
## show
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])












####
## display
def display_image(pg,bxs):

    pg_np = pg.numpy() # use instead of np.array(pg)
    pil_page = Image.fromarray(np.uint8(pg_np * 255))

    boxes = bxs


    # initialize fully transparent overlay
    overlay = Image.new('RGBA', pil_page.size, TINT_COLOR + (0,))
    # initialize "context" for drawing rectangle
    draw = ImageDraw.Draw(overlay)

    # figure
    fig = plt.figure(figsize = (10,10) )
    f_rows = 1
    f_cols = 1
    
    # drawing boxes on overlay
    L = len(boxes)
    for j in range(0,L):
        x, y, x2, y2 = boxes[j]
        shape = [x, y, x2, y2]
        draw.rectangle(shape, fill = TINT_COLOR+(OPACITY,),outline=OUTLINE_COLOR)
    
    new = Image.alpha_composite(pil_page.convert('RGBA'), overlay)

    fig.add_subplot( f_rows, f_cols, 1 )

    plt.tick_params(left=False,bottom=False)
    plt.axis('off')
    plt.imshow(new)












####
## create mask
def create_mask(bounding_box, image):

    im = image.squeeze()
    dims = im.shape
    dims2d = [dims[0], dims[1]]

    # blank 8-bit mask
    mask = Image.new('L', dims2d, 0)

    # initialize draw session
    draw = ImageDraw.Draw(mask)

    # get bb and draw
    bb_np = np.array(bounding_box)
    bb = bb_np.astype(np.int)
    x_1, y_1, x_2, y_2 = bb
    draw.rectangle([(x_1,y_1),(x_2,y_2)], fill = 1)
    
    mask_np = np.array(mask)

    mask_tens = torch.as_tensor(mask_np, dtype = torch.float32)

    mask_bool = mask_tens.ge(1)

    return(mask_bool)

"""
"""
####
#  Dataset class
####
"""
"""


class percolation_papers(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, transforms):

        self.root = dataset_dir
        self.transforms = transforms

        # paths
        self.pages_path = os.path.join(
            self.root, "jpg_pages" )
        self.pdfs_path = os.path.join(
            self.root, "pdf")
        self.labels_path = os.path.join(
            self.root, "labels_xml")
        
        # lists
        self.pages = list(
            sorted(os.listdir(self.pages_path)))
        self.pdfs = list(
            sorted(os.listdir(self.pdfs_path)))
        self.labels = list(
            sorted(os.listdir(self.labels_path)))

    # required
    def __len__(self):
        return(len(self.pages))

    def __getitem__(self, index):
        

        file_name = self.pages[index]
        page_path = os.path.join(self.pages_path, file_name)
        item_name = file_name.split('.')[0]
        auth, year, word, page_number = item_name.split('_')
        xml_file = item_name + '.xml'
        boxes_path = os.path.join(self.labels_path, xml_file)

        # load xml
        # xml = ET.parse(xml_file) <-- susp.
        xml = ET.parse(boxes_path)
        xml_root = xml.getroot()

        # this
        is_empty = xml_root.tag == 'empty'

        # the image itself
        page = read_image(page_path)
        
        # initialize target items
        boxes = []
        masks = []
        labels = []

        if is_empty:
            # prep boxes, masks, labels
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype = torch.int64)


        else:
            # prep boxes
            xmin_Elements = xml_root.findall("./object/bndbox/xmin")
            xmin = [ int(j.text) for j in xmin_Elements ]
            ymin_Elements = xml_root.findall("./object/bndbox/ymin")
            ymin = [ int(j.text) for j in ymin_Elements ]
            xmax_Elements = xml_root.findall("./object/bndbox/xmax")
            xmax = [ int(j.text) for j in xmax_Elements ]
            ymax_Elements = xml_root.findall("./object/bndbox/ymax")
            ymax = [ int(j.text) for j in ymax_Elements ]
            
            # load labels
            labels = torch.ones( len(xmin), dtype = torch.int64)
            
            ## load boxes and recast
            for j in range(0,len(xmin)):
                box = [ xmin[j], ymin[j], xmax[j], ymax[j] ]
                boxes.append(box)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)


            # load masks and recast
            for j in range(0,len(xmin)):
                mask = create_mask(boxes[j], page)
                masks.append(mask)
            #??# masks = torch.as_tensor(masks)


        # initialize dict. to be returned
        target = {}

        # collect last objects to return: image_id,area
        image_id = torch.tensor([index])

        # wrap up target dictionary
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["isempty"] = is_empty

        # transforms
        
        if self.transforms is not None:
            page, target = self.transforms( page, target )
        
        # return
        return page, target


"""
Instead of using the following function, which gives more flexibility in terms of applying a whole sequence of transformations, we just use a switch based on the object passed being the string

            "option1"
            
            transforms.ToPILImage()
"""


# ToTensor() converts PIL or np into FloatTensor,
# and scales pixel intensity values to the range [0,1]

#
def get_transforms():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

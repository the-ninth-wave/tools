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
                               Part I
                               
                   ( displaying the local directory )


Modifying code from: 

https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

The following regex describes the pattern 
we want to use as a criterion for omitting
the listing 
"""
regex_raw_all = r"[/][.]|/_|Icon"
regex_all = re.compile(regex_raw_all)

#key symbols in tree diagram display
element_prefix_middle = '├──'
element_prefix_last = '└──'
parent_prefix_middle = '    '
parent_prefix_last = '│   '

class DisplayablePath(object):
    
    element_prefix_middle = '├──'
    element_prefix_last = '└──'
    parent_prefix_middle = '    '
    parent_prefix_last = '│   '

    # initialize
    def __init__(self, path, parent_path, is_last):
        
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
    
    @property
    def displayname(self):
        
        boo = self.path.is_dir()
        if boo:
            return self.path.name + '/'
        else:
            return self.path.name


    #    make_tree 
    #
    # is a recursive method for 
    # displaying an ASCII tree diagram  
    #
    @classmethod
    def make_tree(cls, 
                  root, 
                  parent=None, 
                  is_last=False, 
                  criteria=None):
        
        root = Path(str(root))   #  init class object for root
        displayable_root = cls(root, parent, is_last)
        yield displayable_root        #   yield \approx "return"
        
        """
        #   'criteria' allows us to filter what's displayed
        #   we want this to be general so that we can more 
        #   smoothly attach other criteria for handling
        #   and displaying different tree structures, like
        #   the ElementTree structures
        
        We set the class default to handle local files below.
        """
        criteria = criteria or cls._default_criteria
        
        
        # collect children of the given current node,
        # filtered by criteria
        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        
        # count = 1 effectively sets tree depth to one.
        count = 1
        
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1


    #     _default_criteria
    #
    # the default criterion is meant to apply
    # to the MacOS local env, so that no
    # superfluous information is displayed in the tree. 
    #
    @classmethod
    def _default_criteria(cls, path):
        
        boo = True 
        path_str = str(path)
        if re.search(regex_all, str(path_str)) != None:
            boo = False
        return boo
        

    @property 
    def displayable(self):
        
        
        if self.parent is None: #    'True' indicates self is root
            return self.displayname #    the return skips the rest
        
        """
        to see what follows this, self must not be the root 
        """
        
        _filename_prefix = (self.element_prefix_last # └──
                            if 
                            self.is_last
                            else 
                            self.element_prefix_middle) # ├──
        
        # we combine the prefix with the file name
        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]
        
        parent = self.parent #    record self's parent
        

        while parent and parent.parent is not None:
            parts.append(self.parent_prefix_middle # (space)
                         if parent.is_last
                         else self.parent_prefix_last) # │   
            parent = parent.parent # this step climbs the tree
            
        
        # Because we append to "parts" as we climb the tree, 
        # we must reverse the sequence built by the climbing          
        return ''.join(reversed(parts))

"""

    disp() the the central 'output' of Part I,
    the point is to visualize the local directory as
    a tree

"""
def disp():
    paths = DisplayablePath.make_tree( os.getcwd() )
    for path in paths:
        print(path.displayable)

        
        
        
        
        
        
        
####################################################
####################################################
"""
                               Part II
                               
        ( analogous methods for xml files, using ElementTree )

The methods display_xml and display_xml_text below are like disp()
for objects in ElementTree. 

Sources:

* https://stackoverflow.com/questions/17275524/xml-etree-elementtree-get-node-depth

* https://stackoverflow.com/questions/323750/how-to-access-the-previous-next-element-in-a-for-loop

* https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node


"""

def display_xml(path):
    
    tree = ET.parse(path)
    root = tree.getroot()
    
    for i in root.iter():
        s = displayable_Element(path, i, i.attrib['is_last'])
        print(s + ' :  ' + i.text)
        
"""
                       Helper functions 
"""

def depth_iter(element, tag = None):
    stack = []
    stack.append(iter([element]))
    
    while stack:
        e = next(stack[-1], None)
        if e == None:
            stack.pop()
        else:
            stack.append(iter(e))
            if tag == None or e.tag == tag:
                yield (e, len(stack) - 1)


def depth_record(tree):
    
    root = tree.getroot()
    
    for j in depth_iter(root):
        j[0].set('depth', str(j[1]))
        
    return(tree)



def last_record(tree):
    
    root = tree.getroot()
    
    # records depth of each element, as they are 
    # visited by depth-first search
    depth_v = [] 
    
    for element in root.iter():
        depth_v += [ int(element.attrib['depth'])]
        
    last_v = []
    
    # looking index ahead inside the for loop
    # so we use the range   [ 0 , len( depth_v ) - 1 ]
    for ind in range(0,len(depth_v)-1):
        boo = False
        if depth_v[ind] > depth_v[ind+1]:
            boo = True
        
        last_v += [ boo ]
       
    last_v += [ True ] # last item always has last status set to true
    
    count = 0 # annotate each tree element with the is_last attrib
    
    for element in root.iter():
        element.set( 'is_last' , str( last_v[count] ) )
        count += 1 
            
    return(tree)

#   parent()
#
# creates a dictionary with ( element, parent ) as 
# ( key, value ) pairs. 
#
def parent(tree, element):
    
    root = tree.getroot()

    parent_map = {c:p for p in tree.iter() for c in p}

    parent_list = []
    for key, value in parent_map.items():
        parent_list += [[ key, value]]  
    
    ind = 0 # get index
    count = 0
    for el in root.iter():
        if el == element:
            ind = count
        count += 1 
    
    return(parent_list[ind-1][1])

"""
    displayable_Element
    
This is a standalone method in part II meant to copy the functionality
of the class property displayable() for DisplayablePath objects. 

We do this here without creating a new class for "displayable 
element trees."


We pass the tree so that we can detect
when the root is passed, via tag - this assumes 
the tag of the root is unique. 

We assume the tree has been annotated so that

    * each Element has an attribute 'depth'
    * each Element has an attribute 'is_last'
    
both attributes are strings, but on use are respectively recast as int and bool. 

To ensure that the xml file to be read in is properly
annotated, we call the next function to do preprocessing.

When playing around, we used a fixed string for the path to the xml file, and so preprocess had no arguments. We had set

    tree = ET.parse('dcd_clips/labels_xml/1.xml')
    root = tree.getroot()

then called

    preprocess()

below preprocess must take in a path, and 
instead of passing the tree defined just above through ET.parse,
we pass the general path and apply the ET.parsse method inside 
the displayable_Element method
"""

def preprocess(path): # e.g.,  path = 'dcd_clips/labels_xml/1.xml'
    
    tree = ET.parse(path)
    root = tree.getroot()
    tree = depth_record(tree) # record depth attribute
    tree = last_record(tree) # record is_last attribute
    root = tree.getroot()
    tree.write(path)

# tree = ET.parse('dcd_clips/labels_xml/1.xml')
# root = tree.getroot()
# preprocess()

def displayable_Element(path, element, is_last):
    
    boo = bool(is_last)
    tree = ET.parse(path)
    root = tree.getroot()
    
    preprocess(path)

    # the text to be displayed 
    display = element.tag 
        
    if root.tag == element.tag: #   if element is root
        return( display ) 

    prefix = (element_prefix_last # └──
                        if 
                        boo
                        else 
                        element_prefix_middle) # ├──
    
    parts = ['{!s} {!s}'.format( prefix, display )]
    
    ancestor = element

    while int(ancestor.attrib['depth']) > 1:
        boo3 = bool(ancestor.attrib['is_last'])
        if boo3:
            parts.append(parent_prefix_middle)
        else:
            parts.append(parent_prefix_last)
        
        ancestor = parent(tree, ancestor)
        
    return ''.join(reversed(parts))

####################################################
####################################################
"""
                               Part III
                               
            ( custom dataset class, display methods )



Sources:

* https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

* https://stackoverflow.com/questions/13887863/extract-bounding-box-and-save-it-as-an-image#13887948

* https://danielcorcoranssql.wordpress.com/2019/04/01/cv2detecting-contours/

* https://pypi.org/project/cv2-tools/



"""

# semicolon suppresses output in jupyter 

#######

"""
reqs:

import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data
"""

from glob import glob

class dcd(torch.utils.data.Dataset):

    def __init__(self, root_dir, transforms, mac):

        self.root_dir = root_dir

        self.transforms = transforms
        
        self.mac = bool(mac)
        
        self.images_path = os.path.join(
            self.root_dir, 
            "dcd_clips/images/"
            )
        
        self.labels_csv_path = os.path.join(
            self.root_dir, 
            "dcd_clips/labels_csv/"
            )
        
        self.labels_xml_path = os.path.join(
            self.root_dir, 
            "dcd_clips/labels_xml/"
            )
        
        if self.mac:
            for file in glob( self.images_path ):
                if regex_all.search(str(file)):
                    os.remove(file)
                    
            for file in glob( self.labels_csv_path ):
                if regex_all.search(str(file)):
                    os.remove(file)
            
            for file in glob( self.labels_xml_path ):
                if regex_all.search(str(file)):
                    os.remove(file)
                    
        
        

        self.images = list(
            sorted( os.listdir( self.images_path ))
        )

        self.labels_csv = list(
            sorted( os.listdir( self.labels_csv_path ))
        )


        self.labels_xml = list(
            sorted( os.listdir( self.labels_xml_path ))
        )
        
        # recall:
        #
        # regex_raw_all = r"[/][.]|/_|Icon"
        # regex_all = re.compile(regex_raw_all)
        
        
        self.images_len = len(self.images)
        
        self.labels_csv_len = len(self.labels_csv)
        
        self.labels_xml_len = len(self.labels_xml)
        
    def __len__(self):
        return(self.images_len)

    def __getitem__(self, index):

        item_images_path = os.path.join(
            self.images_path,
            self.images[index]
            )
        
        item_labels_csv_path = os.path.join(
            self.labels_csv_path,
            self.labels_csv[index]
            )
        
        item_labels_xml_path = os.path.join(
            self.labels_xml_path,
            self.labels_xml[index]
        )
        
        # 
        image = Image.open(item_images_path).convert("RGB")
        image_cv2 = cv2.imread(item_images_path)
        
        """
        A little sloppy here: instead of returning
        two objects, image target pair, which i feel is natural
        I will also return a copy of the image read by
        the cv2 library.
        
        This is to be able to annotate in the display 
        function we build below. 
        """

        xml = ET.parse(item_labels_xml_path)
        xml_root = xml.getroot()

        xmin = xml_root.find("./object/bndbox/xmin").text
        ymin = xml_root.find("./object/bndbox/ymin").text
        xmax = xml_root.find("./object/bndbox/xmax").text
        ymax = xml_root.find("./object/bndbox/ymax").text

        # The x and y coordinates are those (of four possible corners)
        # chosen earliest in lexicographical order
        x = int(xmin)
        y = int(ymin)
        w = int(xmax) - x
        h = int(ymax) - y

        b = [ x, y, w, h ]

        boxes = []
        boxes.append(b)

        boxes = torch.as_tensor(boxes, dtype = torch.float32)
                                
        target = {}
        target["boxes"] = boxes

        return np.array(image), target, image_cv2 
    
    # uses cv2 module / library
    @classmethod
    def display_image_and_target(self,index):
        
        image, target, copy = getitem(index)
        
        # so far, only display first box
        b = target["boxes"][0]
        cv2.rectangle(copy, 
                      (b[0], b[1]), 
                      (b[0]+b[2], b[1]+b[3]), 
                      (0, 255, 0), 
                      2)

        
        # figure
        fig = plt.figure( figsize = (10,10) )
        f_rows = 1 #    formatting 
        f_cols = 2 #    


        fig.add_subplot( f_rows, f_cols, 1 ) # 1st subplot
        plt.imshow(image)
        plt.tick_params(left=False,bottom=False)
        plt.axis('off')

        fig.add_subplot( f_rows, f_cols, 2 ) # 2nd subplot
        plt.imshow(copy)
        plt.tick_params(left=False,bottom=False)
        plt.axis('off')

        plt.show()
        
"""
                Helper: get_transform
reqs:

from engine import train_one_epoch, evaluate ??
import utils
import transforms as T
"""


# def get_transform(train):
    # transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    # transforms.append(T.ToTensor())
    # if train:
        # during training, randomly flip the training images
        # and ground truth for data augmentation
        # transforms.append(T.RandomHorizontalFlip(0.5))
    # return T.Compose(transforms)
    
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
    
   
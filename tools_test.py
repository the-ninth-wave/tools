#
## imports
#
import os
import re
from bs4 import BeautifulSoup # Beautiful Soup
from pathlib import Path
import xml.etree.ElementTree as ET

def test():
    print("Hello world")
    
"""

Modifying code from: 

https://stackoverflow.com/
questions/9727673/list-directory-tree-structure-in-python

"""
###
## req. imports:
#
# from pathlib import Path
# import re
# import xml.etree.ElementTree as ET



"""
The following regex describes the pattern 
we want to use as a criterion for omitting
the listing 
"""
regex_raw_all = r"[/][.]|/_|Icon"
regex_all = re.compile(regex_raw_all)
#
##
#
class DisplayablePath(object):
    
    # key symbols
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    # initialize
    def __init__(self, 
                 path, 
                 parent_path, 
                 is_last):
        
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
    
    """
    displayname
    """
    @property
    def displayname(self):
        
        boo = self.path.is_dir()
        if boo:
            return self.path.name + '/'
        else:
            return self.path.name

    """
    make_tree 
    
        is a recursive method for 
        displaying an ASCII tree diagram
        
    """
    @classmethod
    def make_tree(cls, 
                  root, 
                  parent=None, 
                  is_last=False, 
                  criteria=None):
        
        # initializing a 'class object' for root
        root = Path(str(root))
        displayable_root = cls(root, parent, is_last)
        yield displayable_root        #   yield functions as "return"
        
        """
        #   'criteria' allows us to filter what's displayed
        #   we want this to be general so that we can more 
        #   smoothly attach other criteria for handling
        #   and displaying different tree structures, like
        #   the ElementTree structures
        
        We set the class default to handle local files below.
        """
        #
        # criteria for display
        #
        criteria = criteria or cls._default_criteria
        
        #
        # collect children of the given current node,
        # filtered by criteria
        #
        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        
        #
        # count sets tree depth to one.
        count = 1
        
        #
        # we perform the recursion for each element of children
        #
        """
        
        my understanding of the recursive step is still hazy, 
        especially regarding the distinct types of the "returns" 
        in the two cases below
        
        """
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

    """
    _default_criteria
    
        the default criterion is meant to apply
        to the MacOS local env, so that no
        superfluous information is displayed in the tree. 
        
        
    """
    @classmethod
    def _default_criteria(cls, path):
        
        boo = True 
        path_str = str(path)
        if re.search(regex_all, str(path_str)) != None:
            boo = False
        return boo
        

    """
    displayable
    
        key symbols
    
        display_filename_prefix_middle = '├──'
        display_filename_prefix_last = '└──'
        display_parent_prefix_middle = '    '
        display_parent_prefix_last = '│   '
        
    """
    @property # ??
    def displayable(self):
        
        # this indicates self is the root
        if self.parent is None:
            return self.displayname #    the return skips the rest
        
        ###
        ## to see this, self must not be the root 
        # however, this is the part of the display that 
        # links to the rest of the tree
        #
        _filename_prefix = (self.display_filename_prefix_last # └──
                            if 
                            self.is_last
                            else 
                            self.display_filename_prefix_middle) # ├──
        
        ###
        ## we combine the prefix with the file name into
        # a _single_ string
        #
        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]
        
        ###
        ## record self's parent
        #
        parent = self.parent
        
        ### 
        ## while parent is not root, and 
        #
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle # (space)
                         if parent.is_last
                         else self.display_parent_prefix_last) # │   
            parent = parent.parent # this step climbs the tree
            
            """
            Because we append to "parts"
            as we climb the tree, we must reverse 
            the sequence built by the climbing
            of the tree
            """

        return ''.join(reversed(parts))

"""
Above code from:
https://stackoverflow.com/
questions/9727673/list-directory-tree-structure-in-python
"""
#
##
#

def disp():
    paths = DisplayablePath.make_tree( os.getcwd() )
    for path in paths:
        print(path.displayable)


def display_xml(root):
    for i in root.iter():
        s = displayable_Element( tree, i, i.attrib['is_last'])
        print(s)
        
## returns attributes as a formatted string
# to be printed, as a function of element
import json

def att_string(element):
    return(json.dumps(element.attrib))

def display_xml_text(root):
    for element in root.iter():
    
        # tree
        s = displayable_Element( tree, element, element.attrib['is_last'])
    
        print(s + ' :  ' + element.text)
        
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
"""
One can call depth_iter by

for i in depth_iter(root):
    print(i[0].tag + '\t' + str(i[1]) )

https://stackoverflow.com/questions/17275524/xml-etree-elementtree-get-node-depth

"""

def depth_record(tree):
    
    tree = tree
    
    root = tree.getroot()
    
    for depth_element in depth_iter(root):
        
        # the element itself
        depth_element[0].set( 'depth' , str( depth_element[1]) )
        
    return(tree)

"""
To test this works:

for i in root.iter():
    print(i.attrib['depth'])

Once we have recorded depth, we can use the depth of consecutive elements to determine which are "last"

Rmk

    https://stackoverflow.com/questions/323750/how-to-access-the-previous-next-element-in-a-for-loop

has an answer including a utility function that gives a sliding window on an iterator. I couldn't get this to work, but seems worth knowing about.

"""

import collections, itertools


def last_record(tree):
    
    # store tree
    tree = tree
    
    # store root
    root = tree.getroot()
    
    # this vector records the depth of each element,
    # in the order the elements are visited by iteration,
    # so "depth-first"
    depth_v = []
    
    for element in root.iter():
        depth_v += [ int(element.attrib['depth'])]
        
    
    last_v = []
    
    # we are looking one ahead inside the for loop
    # so we use the range
    """
     0  to  len( depth_v ) - 1
    """
    # in order to not plug in an invalid index
    for ind in range(0,len(depth_v)-1):
        boo = False
        if depth_v[ind] > depth_v[ind+1]:
            boo = True
        
        last_v += [ boo ]
        
    # the last item listed should
    # always have last status set to true.
    
    last_v += [ True ]
    
    # now we annotate each element of the tree with the 
    # "is_last" attribute information
    
    count = 0
    
    for element in root.iter():
        element.set( 'is_last' , str( last_v[count] ) )
        count += 1 
            
    return(tree)

"""
From

    https://stackoverflow.com/questions/2170610/access-elementtree-node-parent-node

I take code for the parent_map; this information needed in the method below.

"""

# This should just return the parent of the given element
def parent(tree, element):
    # from elementTree spellbook;
    # this creates a dictionary of the form
    """
        key     :   value
        element :   parent
    
    """
    parent_map = {c:p for p in tree.iter() for c in p}
    ###
    ## to use, will collect these pairs as an array
    #
    parent_list = []

    for key, value in parent_map.items():
        parent_list += [[ key, value]]
        
    
    # get index
    ind = 0
    count = 0
    for el in root.iter():
        if el == element:
            ind = count
        count += 1 
    
    return(parent_list[ind-1][1])

"""
`displayable_Element`

Instead of trying to pass the data of an xml file
to the class created above, mediated by an ElementTree
object, let's try to repurpose the 
"displayable" method to operate on Element objects.

We also pass the tree so that we can detect
when the root is passed, via tag. 

This assumes the tag of the root is unique. 

Finally, we pass a Boolean indicating whether the element
is last among children of its parent. 

_key symbols_
    
    element_prefix_middle = '├──'
    element_prefix_last = '└──'
    parent_prefix_middle = '    '
    parent_prefix_last = '│   '

Note that we assume the tree has been annotated so that

    * each Element has an attribute 'depth'
    * each Element has an attribute 'is_last'
    
both attributes are strings, but on use are cast as `int` and `Bool` respectively. 

"""

def preprocess():
    # record depth attribute
    tree = ET.parse('dcd_clips/labels_xml/1.xml')
    root = tree.getroot()
    tree = depth_record(tree)
    # record is_last attribute
    tree = last_record(tree)
    root = tree.getroot()
    
    tree.write('dcd_clips/labels_xml/1.xml')

tree = ET.parse('dcd_clips/labels_xml/1.xml')
root = tree.getroot()
#preprocess()

def displayable_Element( tree, element, is_last ):
    boo = bool(is_last)
    
    # key symbols in tree diagram display
    element_prefix_middle = '├──'
    element_prefix_last = '└──'
    parent_prefix_middle = '    '
    parent_prefix_last = '│   '
    
    # the text to be displayed 
    display = element.tag 
        
    # this indicates the element passed is the root
    if tree.getroot().tag == element.tag:
        return( display ) 
        #    the above return skips the rest
        #    
        #    The string in the return plays role of
        #    "self.displayname" from original method
        
    ###
    ## to see this, self must not be the root 
    # however, this is the part of the display that 
    # links to the rest of the tree
    #
    prefix = (element_prefix_last # └──
                        if 
                        boo
                        else 
                        element_prefix_middle) # ├──
        
    ###
    ## combine the diagram prefix with display text
    #
    parts = ['{!s} {!s}'.format( prefix, display )]
        
    ###
    ## get the parent
    #
    ancestor = element

    
    
    while int(ancestor.attrib['depth']) > 1:
        boo3 = bool(ancestor.attrib['is_last'])
        if boo3:
            parts.append(parent_prefix_middle)
        else:
            parts.append(parent_prefix_last)
        
        ancestor = parent(tree, ancestor)
        
    return ''.join(reversed(parts))
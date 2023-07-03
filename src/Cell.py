import numpy as np
import pandas as pd
import xarray
import inspect
import torch

def logger(comment):
    """Function to print Class::Name - Comments from any classes

    Args:
        comment (str): comment to add to the logger
    """
    class_fun_str = inspect.stack()[1][0].f_locals["self"].__class__.__name__ + "::" + \
                    inspect.stack()[1].function + "()"
    print("{:<40} - {:<50}".format(class_fun_str, comment))

class Cell(object):
    def __init__(self, x, loc, label, e=None, A=None, seg=None, mask=None, img=None, pi=None, z=None):
        """This class contain a Cell and all data related to a single-cell c
        
        Args:
            x (torch.tensor, 1D): the spatial gene expression, from dots confidently attributed to cell c
            A (xarray, 3D): 3D spatial expression containing coordinate of gene counts in local neighborhood
            loc (pd.Series, 1D): contain x,y location of cell c centroid, accessible via .loc.x or .loc.y
            seg (np.array, 2D): segmentation centered on cell c containing neighbor cells in local neighborhood
            mask (np.array, 2D): binary mask from segmentated object in local neighborhood
            img (np.array, 2D): image centered on cell c containing neighbor cells in local neighborhood
            label (int): label of cell c
            e (int): size of local neighborhood window, centroids will be enlarged by +/- e 
            pi (torch 1D tensor): probability to be affiliated to each K cell type
            z (torch 1D tensor): assignment of cell type based on pi_c
        """
        self.x = torch.tensor(x)
        self.A = A
        self.loc = loc
        self.label = label
        self.seg = seg
        self.mask = mask
        self.img = img
        self.pi = pi
        self.z = z
        self.x_start = None ; self.x_end = None
        self.y_start = None ; self.y_end = None
        self.on_boarder = False
        self.input_CNN = None
        self.testing = None

    def initialize(self, e, x_width, y_width):
        # record coordinates of local neighborhood
        self.x_start = int(self.loc.x - e) ; self.x_end = int(self.loc.x + e)
        self.y_start = int(self.loc.y - e) ; self.y_end = int(self.loc.y + e)
        # Check if cell is on boarder
        if self.x_start < 0 or self.y_start < 0 or self.x_end > x_width or self.y_end > y_width:
            self.on_boarder = True

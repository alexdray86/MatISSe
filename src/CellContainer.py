import numpy as np
import pandas as pd
import xarray
import inspect
import torch 
import sys
from src.Cell import Cell
from scipy.spatial import KDTree
from skimage.measure import regionprops 

def logger(comment):
    """Function to print Class::Name - Comments from any classes

    Args:
        comment (str): comment to add to the logger
    """
    class_fun_str = inspect.stack()[1][0].f_locals["self"].__class__.__name__ + "::" + \
                    inspect.stack()[1].function + "()"
    print("{:<40} - {:<50}".format(class_fun_str, comment))

def drawProgressBar(percent, barLen = 100):
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

class CellContainer(object):
    def __init__(self, img, seg, D, e, S, A=None, X=None, loc=None, cell_labels=None, gene_names=None, 
                 celltype_labels=None, min_count=5, ignore_cells_with_0transript = True):
        """CellContainer class contains all segmented cells from an image, and their associated spatial gene expression
        
        Args:
            img (np.array, 2D): dapi staining raw image, size (width_X, width_Y)
            seg (np.array, 2D): nuclei segmentation based on dapi staining
            D (pd.DataFrame, 3 columns): Gene dots information, columns are (gene_name, x_coord, y_coord).
            e (int): size of enlargement around cell centroids for making crops used by CNN.
            S (pd.DataFrame): average expression per cell-type derived from a scRNA-seq reference dataset
            A (xarray, 3D): contains the dot corresponding the spatial gene counts.xarray of size (G, width_X, width_Y), G being the number of genes, and (width_X,width_Y) the size of the full image. Calculated on the fly from D.
            X (np.array, 2D): contains the gene counts confidently assigned to each cell c segmented in the spatial data. Calculated on the fly from seg, C, and D.
            loc (pandas, 3 columns): cell centroid information, columns are (cell_label, x_coord, y_coord)
            cell_labels (pd.Series, 1D): labels of cells in X, based on labels in segmentation matrix. Calculated on the fly from seg.
            gene_names (pd.Series, 1D): labels of genes in matrix A, X, and M. Calculated on the fly from unique gene names in D.
            celltype_labels (pd.Series, 1D): labels of cell-types in M. 
            ignore_cells_with_0transript (bool): [default: True] defines if cells that get 0 RNA transcript should be ignored
        """
        self.img = img
        self.seg = seg
        self.D = D
        self.e = e
        self.S = S
        self.S_t = torch.tensor(np.array(S))# / S.sum()))
        self.A = A 
        self.X = X
        self.loc = None
        self.D_closest_cell = None
        self.cell_labels = pd.Series(np.unique(self.seg)[1:]) # Build variable with unique Cell IDs
        self.ignore_cells_with_0transript = ignore_cells_with_0transript
        self.cells = [] # list containing all Cell in image
        self.x_width = self.seg.shape[0]
        self.y_width = self.seg.shape[1]
        self.n_cells_considered = 0
        self.train_test_list = []
        self.min_count = min_count
        # set-up function 
        self.define_gene_overlap()

    def define_gene_overlap(self):
        """Function to find the list of gene names that overlaps between input data matrices"""
        self.gene_names = pd.Series(sorted(set(self.D.name).intersection(set(self.S.index))))
        self.D = self.D.iloc[np.array(self.D.name.isin(self.gene_names))]
        self.D.index = range(self.D.shape[0]) # reindex
        self.S = self.S.iloc[np.array(self.S.index.isin(self.gene_names))]
        self.S.sort_index(inplace=True)
        if len(self.gene_names) > 0:
            logger(f"Found {len(self.gene_names)} overlapping gene names")
        else:
            raise ValueError('No overlap between gene names in D and S')
        
    def get_cell_centroids(self):
        """Function to find cell centroids based on cell segmentation"""
        logger(f'Computing cell centroids based on segmentation file')
        regionprops_out = regionprops(label_image=self.seg)
        labels = [] ; coord_x = [] ; coord_y = []
        for this_seg in regionprops_out:
            labels.append(this_seg.label)
            coord_x.append(this_seg.centroid[0]) 
            coord_y.append(this_seg.centroid[1]) 
        self.loc = pd.DataFrame([labels, coord_x, coord_y], index=['cell_label', 'x', 'y']).T
        
    def find_closest_cell(self, max_shift=50):
        """Given (cell_x, cell_y) cell centroid coordinates, returns the closest cell in a radius of max_shift"""
        logger(f'Attributing gene dots to closest cells')
        dot_positions = np.array(self.D[['x','y']])
        cell_positions = np.array(self.loc[['x','y']])
        # we use KDTree from scipy.spatial
        tree = KDTree(cell_positions)
        # query the k-d tree for the closest points to set1
        distances, indices = tree.query(dot_positions)
        # get the actual closest points
        closest_cells = self.loc.index[indices]
        closest_cells = np.array(closest_cells)
        closest_cells[distances > max_shift] = 0
        self.D_closest_cell = pd.DataFrame([closest_cells, dot_positions[:,0], dot_positions[:,1]], 
                                      index=['closest_cell', 'x', 'y']).T
        
    def build_X_matrix(self, max_shift=30):
        """Based on segmentation and x,y coordinates of a cell, the closest cell is returned
            If no cell is found in a distance of max_shift around the gene dot, 0 will be
            returns meaning no cell attributed"""
        logger("""Build matrix X based on dot attribution""")
        value = np.ones(self.D_closest_cell.shape[0])
        value[np.array(self.D_closest_cell.closest_cell == 0)] = 0
        self.D_closest_cell.insert(3, 'value', value=value)
        self.D_closest_cell.insert(4, 'name', self.D.name)
        self.X = self.D_closest_cell.pivot_table(index='closest_cell', columns='name', values='value',
                                         aggfunc= 'sum').fillna(0).astype(int)
        self.X = self.X.T.iloc[self.X.columns.isin(self.S.index)].T
        # If we want to add lines for cells with 0 counts: 
        self.X = self.X.reindex(np.unique(self.seg)[1:]-1).fillna(0.0)
        #if self.min_count > 0:
        self.X = self.X.iloc[np.array(self.X.sum(axis=1) >= self.min_count)]
        logger('Keeping '+str(self.X.shape[1])+' cells with more than '+str(self.min_count)+' RNA counts')
        self.cell_labels = self.X.index
        self.gene_names = self.X.columns

    def build_Cell_objects(self):
        """Build Cell objects based on cell_labels, X, seg, and cell centroids, used as training data for our model."""
        logger("Build Cell objects based on X, A, seg, and cell centroids")
        logger("Iterating over cells:")
        n_cells = len(self.cell_labels)
        train_test = np.random.choice([0, 1], size=n_cells, p=[.8, .2])
        for cn, c in enumerate(self.cell_labels):
            Cell_obj = Cell(x=self.X.loc[c], loc=self.loc.loc[c], label=c)
            Cell_obj.initialize(e=self.e, x_width=self.x_width, y_width=self.y_width)
            Cell_obj.testing = train_test[cn]
            self.train_test_list.append(train_test[cn])
            #if not Cell_obj.on_boarder:
            self.cells.append(Cell_obj)
            self.n_cells_considered+=1 
            progress = (cn + 1) / n_cells
            drawProgressBar(progress)
        print(' ')
            
    def create_metadata(self):
        """Create a dataframe with metadata from all cells contained in container."""
        logger("Create a dataframe with metadata from all cells contained in container.")
        all_x, all_y, all_test, all_label, all_counts, all_boarders = [], [], [], [], [], []
        for cell in self.cells:
            all_x.append(int(cell.loc.x))
            all_y.append(int(cell.loc.y))
            all_test.append(int(cell.testing))
            all_label.append(int(cell.label))
            all_counts.append(int(torch.sum(cell.x).detach().numpy()))
            all_boarders.append(int(cell.on_boarder))
        self.metadata = pd.DataFrame([all_label, all_test, all_x, all_y, all_counts, all_boarders],
                                      index=['cell_name', 'test_set', 'x', 'y', 'total_counts', 'on_boarder']).T


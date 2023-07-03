import numpy as np
import pandas as pd
import tifffile
import imageio
import xarray
import scipy
import inspect # from package inspect-it
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
import skimage
import random
from scipy import stats
from PIL import Image
from src.Cell import Cell
from src.CellContainer import CellContainer
from src.FullNet import FullNet
#Image.MAX_IMAGE_PIXELS = None # remove pixel size limit

def drawProgressBar(percent, barLen = 100):
    # percent float from 0 to 1.
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

## Define out of class functions
def build_input_CNN(D_, seg_, img_, gene_names_, cell_, e):
    """Build matrix A based on gene dots in D"""
    #logger("Build matrix A based on gene dots in D")
    #D_.insert(3, 'value_', 1)
    D_['y'] = D_['y'] - cell_.y_start
    D_['x'] = D_['x'] - cell_.x_start
    all_As = np.zeros((187,e*2,e*2))
    for g, gene in enumerate(gene_names_):
        gene_count_1gene = D_.iloc[np.array(D_.name == gene)]
        for i in range(gene_count_1gene.shape[0]):
            x=gene_count_1gene.iloc[i]['x']
            y=gene_count_1gene.iloc[i]['y']
            all_As[g,x,y] += 1
    mask_ = seg_.copy()
    mask_[mask_ > 0] = 1
    # 3D tensor used as input for CNN 
    input_CNN = torch.tensor(np.concatenate(( mask_.reshape((1,) + mask_.shape), 
                                              img_.reshape((1,) + img_.shape), 
                                              all_As )))
    return(input_CNN)

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )
    if clip:
        x = np.clip(x,0,1)
    return x

def normalize_img(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""
    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def launch_training(model, CellCon_obj, learning_rate, n_cells, num_epochs, tau, n_cells_per_batch, window_radius):
    # initialize variables 
    S_n = torch.tensor(np.array(S / S.sum()))
    cell_type_sizes = CellCon_obj.S.sum()
    prev_loss = 0.0
    
    ### Launch the training of the model ### 
    # Create arrays to store results
    all_x_hats = np.zeros((len(CellCon_obj.gene_names), n_cells_per_batch, num_epochs))
    all_pis =  np.zeros((S.shape[1], n_cells_per_batch, num_epochs))
    all_zs = np.zeros((S.shape[1], n_cells_per_batch, num_epochs))
    all_loss = [] #np.zeros((n_cells_per_batch, num_epochs))
    all_rank_average, all_accuracy = [], []

    # Launch the training with num_epochs Epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}, T={tau}, iterating over cells...')
        ### Prepare input/output data ###
        print('Build Mini-Batch for epoch '+str(epoch)+' with '+str(n_cells_per_batch)+' cells')
        # Prepare CNN for a mini-batch of size n_cells_per_batch
        input_tensor = torch.tensor(np.zeros(shape=(n_cells_per_batch,189,window_radius*2,window_radius*2)) )
        output_tensor = torch.tensor(np.zeros(shape=(n_cells_per_batch,187)))
        all_min_z = np.zeros((S_tensor.shape[1], n_cells_per_batch))
        all_min_loss = np.zeros(n_cells_per_batch)
        cell_subset = random.sample(CellCon_obj.cells, n_cells_per_batch)
        for c, cell in enumerate(cell_subset):
            # For now, we ignore cells on the boarder of the image
            if cell.on_boarder: 
                continue
            # prepare input data
            x_true = cell.x.clone().detach().double()
            D_ = CellCon_obj.D
            sel_window = (D_.y > cell.y_start) & (D_.y < cell.y_end) & (D_.x > cell.x_start) & (D_.x < cell.x_end)
            D_sub = D_.iloc[np.array(sel_window)].copy()
            seg_ = CellCon_obj.seg[cell.x_start:cell.x_end,cell.y_start:cell.y_end]
            img_ = CellCon_obj.img[cell.x_start:cell.x_end,cell.y_start:cell.y_end]
            input_cnn = build_input_CNN(D_sub, seg_, img_, CellCon_obj.gene_names, cell, window_radius)
            # Store CNN input/output in tensors 
            input_tensor[c,:,:,:] = input_cnn
            output_tensor[c,:] = x_true
            # Store "ground-truth": find minimum loss for pure cell-type
            min_loss = 100.0
            for k in range(S_n.shape[1]):
                loss_k = criterion(S_n[:,k], x_true)
                if loss_k.item() < min_loss:
                    min_loss = loss_k.item()
                    z_kc = np.zeros(S_n.shape[1])
                    z_kc[k] = 1
            all_min_loss[c] = min_loss ; all_min_z[:, c] = z_kc 
            # Draw progress bar 
            progress = (c + 1) / n_cells_per_batch
            drawProgressBar(progress)
        
        # Standardize input tensor
        # First we standardize gene expression data all together
        means = input_tensor[:,2:,:,:].mean(dim=(1,2,3), keepdim=True)
        stds = input_tensor[:,2:,:,:].std(dim=(1,2,3), keepdim=True)
        input_tensor_n = input_tensor.clone()
        input_tensor_n[:,2:,:,:] = (input_tensor[:,2:,:,:] - means) / stds
        # Then we standardize image/seg separately
        input_tensor_n[:,0,:,:] = (input_tensor_n[:,0,:,:] - input_tensor_n[:,0,:,:].mean())/(input_tensor_n[:,0,:,:].std())
        input_tensor_n[:,1,:,:] = (input_tensor_n[:,1,:,:] - input_tensor_n[:,1,:,:].mean())/(input_tensor_n[:,1,:,:].std()) 
        
        # send output to device
        output_tensor=output_tensor.to(device)

        ### Launch Training ###
        # forward pass
        mu_k, pi_kc, z_kc = model(input_tensor_n.to(device), S_n.to(device), tau)
        
        # L2 regularization
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        
        # Calculating loss 
        loss=criterion(mu_k, output_tensor.T)
        loss_no_reg = loss.item()
        loss = loss + l2_lambda*l2_norm
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # Summary
        print(f'Epoch {epoch+1}/{num_epochs}, Mean Loss = {loss_no_reg:.4f}, Mean Loss with reg = {loss.item():.4f}, best min loss = {all_min_loss.mean()}')
        rank_l = [] ; tp_count = 0
        for i in range(n_cells_per_batch):
            rank_l.append(pd.Series(z_kc[i,:].cpu().detach().numpy()).rank(ascending=False)[np.argmax(all_min_z[:,i])])
            if rank_l[i] == 1:
                tp_count+=1
        print('Mean rank of true cell type: '+str(np.mean(rank_l))+'/16')
        all_rank_average.append(np.mean(rank_l))
        print('Accuracy: '+str(tp_count/n_cells_per_batch))
        all_accuracy.append(tp_count/n_cells_per_batch)
        all_loss.append(loss_no_reg)       
 
        # Draw progress bar 
        progress = (epoch + 1) / num_epochs
        drawProgressBar(progress)

        # Debug
        # check gradient values - control for vanishing gradients
        for name, param in model.named_parameters(): # Check gradient sum 
            print(name, param.grad.abs().sum())

    return all_loss, pi_kc, z_kc, all_rank_average, all_accuracy, model

def launch_testing(model, CellCon_obj, learning_rate, n_cells, num_epochs, tau, n_cells_per_batch, window_radius):
    # iterate over cells
    total_cell_number = len(CellCon_obj.cells)
    total_batch_number = int(total_cell_number/n_cells_per_batch)+1
    all_zs, all_pis = np.zeros((total_cell_number, 16)), np.zeros((total_cell_number, 16))
    all_accuracy, all_ranks, all_loss = np.zeros(total_cell_number), np.zeros(total_cell_number), np.zeros(total_cell_number)
    for i in range(total_batch_number):
        print('cell subset '+str(i)+'/'+str(total_batch_number))
        first_cell = i*n_cells_per_batch
        last_cell  = min((i+1)*n_cells_per_batch, total_cell_number)
        cell_subset = CellCon_obj.cells[first_cell:last_cell]
        print(str(first_cell)+':'+str(last_cell))
        
        # Prepare CNN for each cells 
        input_tensor = torch.tensor(np.zeros(shape=(n_cells_per_batch,189,window_radius*2,window_radius*2)) )
        output_tensor = torch.tensor(np.zeros(shape=(n_cells_per_batch,187)))
        all_min_z = np.zeros((S_tensor.shape[1], n_cells_per_batch))
        all_min_loss = np.zeros(n_cells_per_batch)
        for c, cell in enumerate(cell_subset):
            # prepare input data
            x_true = cell.x.clone().detach().double()
            D_ = CellCon_obj.D
            sel_window = (D_.y > cell.y_start) & (D_.y < cell.y_end) & (D_.x > cell.x_start) & (D_.x < cell.x_end)
            D_sub = D_.iloc[np.array(sel_window)].copy()
            seg_ = CellCon_obj.seg[cell.x_start:cell.x_end,cell.y_start:cell.y_end]
            img_ = CellCon_obj.img[cell.x_start:cell.x_end,cell.y_start:cell.y_end]
            input_cnn = build_input_CNN(D_sub, seg_, img_, CellCon_obj.gene_names, cell, window_radius)
            # Store CNN input in a dictionnary
            input_tensor[c,:,:,:] = input_cnn
            output_tensor[c,:] = x_true
            progress = (c + 1) / n_cells_per_batch
            drawProgressBar(progress)
            # find minimum loss pure cell-type
            min_loss = 100.0
            for k in range(S_n.shape[1]):
                loss_k = criterion(S_n[:,k], x_true)
                if loss_k.item() < min_loss:
                    min_loss = loss_k.item()
                    z_kc = np.zeros(S_n.shape[1])
                    z_kc[k] = 1
            all_min_loss[c] = min_loss
            all_min_z[:, c] = z_kc

        # Standardize input tensors 
        means = input_tensor[:,2:,:,:].mean(dim=(1,2,3), keepdim=True)
        stds = input_tensor[:,2:,:,:].std(dim=(1,2,3), keepdim=True)
        input_tensor_n = input_tensor.clone()
        input_tensor_n[:,2:,:,:] = (input_tensor[:,2:,:,:] - means) / stds
        input_tensor_n[:,0,:,:] = (input_tensor_n[:,0,:,:] - input_tensor_n[:,0,:,:].mean())/(input_tensor_n[:,0,:,:].std())
        input_tensor_n[:,1,:,:] = (input_tensor_n[:,1,:,:] - input_tensor_n[:,1,:,:].mean())/(input_tensor_n[:,1,:,:].std())

        # forward pass
        mu_k, pi_kc, z_kc = model(input_tensor_n.to(device), S_n.to(device), tau)
        
        # Computing loss 
        loss=criterion(mu_k, output_tensor.T)
        
        # Summary
        print(f'Prediction of mini-batch {i+1}/{total_batch_number}, Mean Loss = {loss.item():.4f}, best min loss = {all_min_loss.mean()}')
        rank_l = [] ; tp_count = 0
        for i in range(n_cells_per_batch):
            rank_l.append(pd.Series(z_kc[i,:].detach().numpy()).rank(ascending=False)[np.argmax(all_min_z[:,i])])
            if rank_l[i] == 1:
                tp_count+=1
        print('Mean rank of true cell type: '+str(np.mean(rank_l))+'/16')
        print('Accuracy: '+str(tp_count/n_cells_per_batch))
        
        # Save results
        all_zs[first_cell:last_cell,:] = z_kc.cpu().detach().clone()
        all_pis[first_cell:last_cell,:] = pi_kc.cpu().detach().clone()
        all_accuracy[first_cell:last_cell] = tp_count/n_cells_per_batch
        all_ranks[first_cell:last_cell] = np.mean(rank_l)
        all_loss[first_cell:last_cell] = loss.item() 
    return all_zs, all_pis, all_accuracy, all_ranks, all_loss

if __name__ == "__main__":
    # Define variables 
    Image.MAX_IMAGE_PIXELS = None
    learning_rate = 0.0005
    num_epochs = 2
    tau=0.5
    n_cells_per_batch = 128
    window_radius = 64

    ### Dataset 1 : ... ###
    D = pd.read_csv('data/toni/e13_5_GLM171wt_e10_25_FB_14k_ca-ilp_768_7_11_reformat.csv')
    img = tifffile.imread('data/toni/e13_5_GLM171wt_e19_25_FB14k_Nuclei.tif').T.astype('double')
    seg = imageio.v2.imread('data/toni/e13_5_GLM171wt_e19_25_FB14k_Nuclei_cp_masks.png').T
    S = pd.read_csv("data/scrna_muX_clust16_TaxonomyRank3.csv", index_col='Gene')
    S_tensor = torch.tensor(np.array(S)).double() 
    
    ### Normalize full image ###
    img_n = normalize_img(img)

    ### Generate CellContainer object container Cell objects related to segmented cells in dapi image ###
    CellContainer_obj = CellContainer(img=img_n.copy(), seg=seg.copy(), D=D.copy(), e=window_radius, S=S.copy(),
                                      ignore_cells_with_0transript=True)
    CellContainer_obj.get_cell_centroids()
    CellContainer_obj.find_closest_cell()
    CellContainer_obj.build_X_matrix()
    CellContainer_obj.build_Cell_objects()
    CellContainer_obj.create_metadata()    
    n_cells = CellContainer_obj.n_cells_considered

    ### Build the model ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FullNet(n_genes=CellContainer_obj.S.shape[0], 
                    latent_variable_dim=CellContainer_obj.S.shape[1], 
                    e=window_radius, batch_size=n_cells_per_batch).to(device).double()
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    ### Run the model on the training set - subset of cells with high RNA counts ###
    all_loss, pi_kc, z_kc, all_rank_average, all_accuracy, trained_model = launch_training(model, CellCon_obj=CellContainer_obj, learning_rate=learning_rate, n_cells=CellContainer_obj.n_cells_considered, num_epochs=num_epochs, tau=tau, n_cells_per_batch=n_cells_per_batch, window_radius=window_radius)

    ### Use the model to predict cell-type on a full image ###
    all_zs, all_pis, all_accuracy, all_ranks, all_loss = launch_testing(trained_model, CellCon_obj=CellContainer_obj, learning_rate=learning_rate, n_cells=CellContainer_obj.n_cells_considered, num_epochs=num_epochs, tau=tau, n_cells_per_batch=n_cells_per_batch, window_radius=window_radius)
    
    pd.DataFrame(all_pis).to_csv("results/toniCrop1_t100lr4T01_pis.csv")
    pd.DataFrame(all_zs).to_csv("results/toniCrop1_t100lr4T01_zs.csv")
    pd.DataFrame(all_accuracy).to_csv("results/toniCrop1_t100lr4T01_accuracy.csv")
    pd.DataFrame(all_ranks).to_csv("results/toniCrop1_t100lr4T01_ranks.csv")
    pd.DataFrame(all_loss).to_csv("results/toniCrop1_t100lr4T01_loss.csv")



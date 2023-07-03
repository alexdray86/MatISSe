import numpy as np
import pandas as pd
import xarray
import inspect # from package inspect-it
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU

class FullNet(nn.Module):
    def __init__(self, n_genes, latent_variable_dim, e=64, size_pooling1=4, size_pooling2=2,
                 batch_size=3):
        """
        Model based on nn.Module, which combines : a CNN to produce a feature vector from spatial expression data, 
        which is then combined with x_gc, the gene expression confidently attributed to a cell c. Both vectors are
        then concatenated and passed in a fully connected NN to compute pi_kc, the cell-type probabilities. These
        probability are then converted to a quasi-categorical variable with a gumbel softmax. Using a scRNA-seq as
        a reference, we then compute x_hat with poisson sampling. 
        Args:
            n_genes (int): The number of genes in our spatial expression data. Helps to define the number of channels in CNN
            latent_variable_dim (int): correspond to the number of cell-types we consider, and to the dimension of pi_kc
            tau (float): temperature parameter for gumbel softmax quasi discrete sampling
        """
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_channels = int(n_genes + 2) # we add two channels for img+seg
        self.latent_variable_dim = int(latent_variable_dim)
        self.len_vec_cnn = int(((e*2)*(e*2)*self.n_channels)/((size_pooling1**2)*(size_pooling2**2)))
        self.conv1 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(int(self.n_channels))
        self.leakyrelu1 = nn.LeakyReLU(int(self.n_channels))
        self.conv3 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(int(self.n_channels))
        self.leakyrelu2 = nn.LeakyReLU(int(self.n_channels))
        self.pool1 = nn.MaxPool2d(size_pooling1, size_pooling1) #   self.pool = nn.MaxPool2d(4,4)
        self.pool2 = nn.MaxPool2d(size_pooling2, size_pooling2)
        self.fc1 = nn.Linear(self.len_vec_cnn, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, self.latent_variable_dim)
        self.batch_size = batch_size
        
    def f_lambda(self, input_cnn):
        """
        f_lambda is a function that use two rounds of Convolution/ReLu/MaxPool on a 3D matrix containing dapi 
        image and segmentation, coupled with spatial gene expression. 
        Args:
            input_CNN (3D tensor): contains dapi info and binned spatial expresion
        Returns:
            (1D tensor): a one-dimensional tensor summarizing the image and expression data after CNN 
        """
        residual = input_cnn.clone()
        x = F.relu(self.bnorm1(self.conv1(input_cnn)))
        x = F.relu(self.bnorm1(self.conv2(x)))
        x = torch.add(x, residual)
        x = self.pool1(x)
        residual2 = x.clone()
        x = F.relu(self.bnorm2(self.conv3(x)))
        x = F.relu(self.bnorm2(self.conv4(x)))
        x = torch.add(x, residual2)
        x = self.pool2(x)
        x = x.reshape(self.batch_size, self.len_vec_cnn)
        return(x)

    def f_gamma(self, output_cnn):
        """
        f_gamma takes as input the output of f_lambda, that is to say the 1D tensor output from a CNN, and
        concatenate it with x_gc, another 1D tensor containing the spatial gene expression confidently attributed
        to cell c by proximity. 
        Args:
            output_CNN (1D tensor): a one-dimensional tensor summarizing the image and expression data after CNN 
            x_gc (1D tensor): a one-dimensional tensor containing spatial gene expression counts attributed to a cell
        Returns:
            (1D tensor): a one-dimensional tensor of length K (number of cell-types), corresponding to pi_kc
        """
        # use only CNN input 
        x = F.relu(self.fc1(output_cnn))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def categorical(self, pi_kc, tau):
        """
        Uses a Gumbel softmax to produce a quasi discrete sampling from pi_kc probabilities 
        Args:
            pi_kc (1D tensor): cell type probability distribution
            tau (float): temperature parameter used by gumbel softmax method
        Returns:
            z_kc (1D tensor): quasi one-hot-encoded cell type distribution
        """
        return F.gumbel_softmax(pi_kc, tau=tau, dim=1).double() #
        
    def project_to_centroids(self, z_kc, S):
        """
        Projects a given (quasi) one-hot-encoded cell type distribution into the corresponding cluster centroid
        Args:
            z_kc (1D tensor): quasi one-hot-encoded cell type distribution
            S (2D tensor): Average expression, (genes x cell-types), from a scRNA-seq data in the same context
        Returns:
            mu_k: average expression of cell-type k
        """
        mu_k = torch.matmul(S, z_kc.T)
        return(mu_k)

    def sample_poisson(self, mu_k):
        """
        return a poisson sampling - thus discrete counts attributed to each gene, mimicking x_gc, 
        and using values in mu_k as factor lambda for poisson distribution 
        Args:
            mu_k (1D tensor): Poisson parameter given by the network
        Returns:
            x_hat (1D tensor): the predicted gene expression
        """
        return torch.poisson(mu_k)[:,0]

    def forward(self, input_cnn, S, tau):
        """
        Forward pass of the network
        Args:
            input_CNN (3D tensor): contains dapi info and binned spatial expresion
            x_gc (1D tensor): a one-dimensional tensor containing spatial gene expression counts
            S (2D tensor): average expression per cell-type derived from a scRNA-seq reference dataset
        Returns:
            xhat (1D tensor): the predicted gene expression
        """
        output_cnn = self.f_lambda(input_cnn)
        pi_kc = self.f_gamma(output_cnn)
        z_kc = self.categorical(pi_kc, tau)
        mu_k = self.project_to_centroids(z_kc, S)
        return mu_k, pi_kc, z_kc

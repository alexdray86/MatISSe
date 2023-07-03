import torch
import torch.nn as nn

class FullNet(nn.Module):
    def __init__(self, sc_data):
        """_summary_

        Args:
            sc_data (_type_): _description_
        """
        super().__init__()
        self.sc_data = sc_data

    def f_lambda(self, A, dapi):
        """_summary_

        Args:
            A (_type_): multiome image (A_yxg)
            dapi (_type_): 2-channel stained nuclei image. Channel 0 is
                           actual nuclei stained, channel 1 is location of
                           the cell of interest as a segmentation mask

        Returns:
            _type_: _description_
        """
        raise NotImplementedError


    def f_gamma(self, img_info, X):
        """_summary_

        Args:
            img_info (_type_): multiome image (A_yxg)
            X (_type_): 2-channel stained nuclei image. Channel 0 is
                           actual nuclei stained, channel 1 is location of
                           the cell of interest as a segmentation mask

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

    def project_to_centroids(self, z_kc):
        """Projects a given (quasi) one-hot-encoded cell type
           distribution into the corresponding cluster centroid

        Args:
            z_kc (_type_): quasi one-hot-encoded cell type distribution

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

    def sample_poisson(self, mu_c):
        """_summary_

        Args:
            mu_c (_type_): Poisson parameter given by the network

        Returns:
            _type_: _description_
        """
        raise NotImplementedError

    def forward(self, X, A, dapi):
        """Forward pass of the network

        Args:
            X (_type_): segmented cell in spatial data (X_gc)
            A (_type_): multiome image (A_yxg)
            dapi (_type_): 2-channel stained nuclei image. Channel 0 is
                           actual nuclei stained, channel 1 is location of
                           the cell of interest as a segmentation mask

        Returns:
            _type_: _description_
        """
        img_info = self.f_lambda(A, dapi)
        pi_kc = self.f_gamma(img_info, X)
        z_kc = self.gumbel_softmax(pi_kc, temperature=.5) # !
        mu_c = self.project_to_centroids(z_kc)
        X_hat = self.sample_poisson(mu_c)
        return X_hat

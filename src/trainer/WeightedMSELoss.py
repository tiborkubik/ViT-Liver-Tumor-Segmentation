"""
    :filename WeightedMSELoss.py

    :brief Classical Mean Squared Error loss function with modifiable weights for liver and tumor channels.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""
import torch.nn as nn


class WeightedMSELoss(nn.Module):

    def __init__(self, w_liver=1.0, w_tumor=1.0):
        super().__init__()

        self.w_liver = w_liver
        self.w_tumor = w_tumor

        self.mse = nn.MSELoss()

    def forward(self, x, y):
        w_loss_liver = self.w_liver * self.mse(x[:, 0:1, :], y[:, 0:1, :])
        w_loss_tumor = self.w_tumor * self.mse(x[:, 1:2, :], y[:, 1:2, :])

        return w_loss_liver + w_loss_tumor

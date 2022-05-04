"""
    :filename DiceLoss.py

    :brief Dice Loss for medical image segmentation.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        dims = (1, 2, 3)
        intersection = (inputs * targets).sum(axis=dims)
        dice = (2. * intersection + smooth) / (inputs.sum(axis=dims) + targets.sum(axis=dims) + smooth)

        return torch.mean(1 - dice)

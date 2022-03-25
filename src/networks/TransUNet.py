"""
    :filename TransUNet.py

    :brief TransUNet architecture.

    The original TransUNet paper: https://arxiv.org/abs/2102.04306.

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
import numpy as np
from torch import nn
import src.networks.vit_seg.vit_seg_modeling as vit_seg


class TransUNet(nn.Module):
    """
        TransUNet architecure based on ViT transformer and resnet in encoder part, decoder CUP with skip connections,
        more details about architecture -
        ViT - https://arxiv.org/abs/2010.11929,
        TransUNet - https://arxiv.org/abs/2102.04306.
    """

    def __init__(self, vit_model,
                 n_classes, img_size, weights,
                 vit_patches_size):
        super(TransUNet, self).__init__()
        config_vit = vit_seg.CONFIGS[vit_model]
        config_vit.n_classes = n_classes
        if vit_model.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.net = vit_seg.VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()
        self.net.load_from(weights=np.load(weights))

    def forward(self, image):
        return self.net.forward(image)


if __name__ == '__main__':
    network = TransUNet(vit_model='R50-ViT-B_16',
                        n_classes=3, img_size=224,
                        weights='models/vit_checkpoint/imagenet21k/imagenet21k_R50+ViT-B_16.npz',
                        vit_patches_size=16).to('cuda:0')

    input_tensor = torch.rand(1, 1, 224, 224).to('cuda:0')

    output_tensor = network(input_tensor)

    print(f'Output tensor size: {output_tensor.shape}.')

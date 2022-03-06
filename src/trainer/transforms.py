"""
    :filename transforms.py

    :brief Custom transformations for dataloading pipeline.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""

import elasticdeform
from torchvision import transforms as T
import torch


class ElasticTransform(torch.nn.Module):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
       .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
            Convolutional Neural Networks applied to Visual Document Analysis", in
            Proc. of the International Conference on Document Analysis and
            Recognition, 2003.

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
       """

    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def __call__(self, image):
        return elasticdeform.deform_grid(image, self.displacement)

    def __repr__(self):
        return "Elastic transform"


class Invert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, image):
        return T.functional.invert(image)

    def __repr__(self):
        return "Inverts image"

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

from torchvision import transforms as T
import torch
import numpy as np
import numbers
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
    Adopted from https://github.com/gatsby2016/Augmentation-PyTorch-Transforms.
    Args:
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
        mask (PIL Image) in __call__, if not assign, set None.
    """

    def __init__(self, alpha, sigma):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        return img

    def __call__(self, img):
        return self.RandomElasticCV2(img[..., np.newaxis], self.alpha, self.sigma)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string


class Invert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, image):
        return T.functional.invert(image)

    def __repr__(self):
        return "Inverts image"

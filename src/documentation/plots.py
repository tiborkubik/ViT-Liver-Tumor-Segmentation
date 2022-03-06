"""
    :filename plots.py

    :brief Sample plot function for documentation figures generation.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""

import matplotlib.pyplot as plt


def plot_slice_sample(ct_slice, normalized_slice, mask, savefig=None):
    """
    Plot figure with slice, windowed slice and mask from dataset.
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

    ax1.imshow(ct_slice, cmap='bone')
    ax1.set_title('Original Image')

    ax2.imshow(normalized_slice, cmap='bone')
    ax2.set_title('Windowed Image')

    tumor = mask == 2.0
    ax4.imshow(ct_slice, cmap='bone')
    ax4.imshow(tumor, cmap='Reds', alpha=0.5)
    ax4.set_title('Masked tumor')

    mask[mask == 2.0] = 1.0

    ax3.imshow(ct_slice, cmap='bone')
    ax3.imshow(mask, cmap='Greens', alpha=0.5)
    ax3.set_title('Masked liver')

    if savefig:
        fig.savefig(savefig)
    else:
        fig.show()


def augmentation_diff(pre_img, post_img, pre_mask, post_mask, savefig=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

    ax1.imshow(pre_img, cmap='bone')
    ax1.set_title('Original Image')

    ax2.imshow(post_img, cmap='bone')
    ax2.set_title('Augmented Image')

    ax3.imshow(pre_mask, cmap='bone')
    ax3.set_title('Original Mask')

    ax4.imshow(post_mask, cmap='bone')
    ax4.set_title('Augmented Mask')

    if savefig:
        fig.savefig(savefig)
    else:
        fig.show()

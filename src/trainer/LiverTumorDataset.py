"""
    :filename LiverTumorDataset.py

    :brief Dataset class for segmentation of liver and tumor from CT scans of abdomen.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""

import os
import cv2
import glob
import torch
import logging
import matplotlib
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Subset, random_split, DataLoader

matplotlib.rcParams["figure.dpi"] = 400

EXT = '*.png'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class LiverTumorDataset(Dataset):

    def __init__(self, dataset_path, transforms=None):

        self.dataset_path = dataset_path
        self.transforms = transforms

        self.slices = []  # One sample: tuple (slice .png path, mask .png path).

        w_path = self.dataset_path + 'vols-2d/'

        all_slice_files = [file
                           for path, subdir, files in os.walk(w_path)
                           for file in glob.glob(os.path.join(path, EXT))]

        for slice_path in all_slice_files:
            mask_path = slice_path.replace('volume', 'segmentation').replace('vols', 'segs')
            self.slices.append((slice_path, mask_path))

    def __getitem__(self, item):

        slice = cv2.imread(self.slices[item][0], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.slices[item][1], cv2.IMREAD_GRAYSCALE)

        sample = {
            'image': slice,
            'mask': mask
        }

        if self.transforms is not None:
            for key in sample:
                sample[key] = transforms.ToTensor()(sample[key])

            # TODO: apply custom transformations from self.transforms (specified by argument).
            # sample = self.transforms(sample)

        for key in sample:
            assert sample[key] is not None, \
                f'Invalid {key} in sample {self.slices[item]}'

        return sample

    def __len__(self):

        return len(self.slices)


def get_dataset_loaders(dataset_dir, transforms=None, batch_size=32, workers=1,
                        validation_split=.2, random=True, ddp=False):
    """
    Method prepares torch dataset loaders for the training.

    It performs several steps, which are basically the whole data-prep pipeline:
        -   It creates an instance of LiverTumorDataset. This means that the paths to individual slices from available
            nii volumes are stored in memory. Whole dataset has ~30GB, so the actual image loading is not performed
            directly, but rather in the __getitem__ method when necessary in training loop.
        -   It randomly splits the available training data into training and validation part. Please note that the
            evaluation is completely separate and a custom loader is available for that part.
            The thing is that we want to perform testing on completely separate volumes of liver.
        -   It creates the sampler instances (if needed) and the loaders itself.
            The returned loaders are inherited from torch Data Loaders, so they can be directly used in training loop.

    :param dataset_dir: Path to the dataset of TRAINING+VAL part only!
    :param transforms: Augmentation to be applied during training. Default is None.
    :param batch_size: Training batch size.
    :param workers: Number of workers for parallel data loading. Depends on the number of CPUs available.
    :param validation_split: From range [0.0, 1.0] -> The percentage of data for validation. Default is 0.2.
    :param random: Randomize choice of train/val split.
    :param ddp: Enable when multi-gpu training.

    :return: Train and validation loader.
    """
    train_split = 1 - validation_split

    dataset = LiverTumorDataset(dataset_path=dataset_dir, transforms=transforms)

    train_len = int(train_split * len(dataset))
    val_len = int(validation_split * len(dataset))

    logging.info(f'Length of whole dataset: {len(dataset)}.')
    logging.info(f'With the splitting of {validation_split}:')
    logging.info(f'Length of training set: {train_len}, length of validation set: {val_len}.')

    if random:
        difference = len(dataset) - train_len - val_len  # Just stupid workaround for random splitting and rounding :).
        train_ds, val_ds = random_split(dataset,
                                        [train_len + difference, val_len],
                                        generator=torch.Generator().manual_seed(42))
    else:
        train_ds = Subset(dataset, np.arange(0, train_len))
        val_ds = Subset(dataset, np.arange(train_len, train_len + val_len))

    train_sampler = DistributedSampler(train_ds) if ddp else None
    val_sampler = DistributedSampler(val_ds) if ddp else None

    loader_kwargs = {'batch_size': batch_size, 'num_workers': workers, 'pin_memory': True, 'drop_last': True}
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)

    if batch_size is not None:
        if len(train_loader.dataset.indices) < batch_size:
            logging.warning(f'Training data subset too small: {len(train_loader.dataset.indices)}.')

        if len(val_loader.dataset.indices) < batch_size:
            logging.warning(f'Validation data subset too small: {len(val_loader.dataset.indices)}.')

    return train_loader, val_loader


def pre_process_niis(path):
    """
    Use this to extract slices and corresponding masks from .nii volumes as png images and store them on disk.

    :param path: Path to the root of dataset folder.

    :return: void
    """
    w_path = path + 'vols-3d/'

    # Paths of .nii volumes.
    all_nii_files = [file
                     for path, subdir, files in os.walk(w_path)
                     for file in glob.glob(os.path.join(path, '*.nii'))]

    for nii_file in all_nii_files:
        vol = nib.load(nii_file)
        vol = vol.get_fdata()
        num_slices = vol.shape[2]  # The shape is denoted e.g (512, 512, 826), where the last one is num of slices.

        vol_name = nii_file.split(".")[-2].split('/')[-1]

        for i in range(num_slices):
            cv2.imwrite(path + f'vols-2d/{vol_name}-{i}.png', vol[:, :, i])

        seg_nii_path = nii_file.replace('volume', 'segmentation').replace('vols', 'segs')
        seg = nib.load(seg_nii_path)
        seg = seg.get_fdata()

        seg_name = seg_nii_path.split(".")[-2].split('/')[-1]

        for i in range(num_slices):
            cv2.imwrite(path + f'segs-2d/{seg_name}-{i}.png', seg[:, :, i])


if __name__ == '__main__':
    ''' Uncomment this to extract the individual slices and masks as png images (from nii volumes). '''
    # pre_process_niis('../../data/train-val/')

    ''' So this method is basically ready-to-use in training pipeline. Here, it's just for testing. '''
    test_batch_size = 8
    train_loader, val_loader = get_dataset_loaders('../../data/train-val/', batch_size=test_batch_size)

    sample = next(iter(train_loader))
    #
    # f, axarr = plt.subplots(2, test_batch_size)
    # f.set_size_inches(10, 3)
    # for i in range(test_batch_size):
    #     axarr[0, i].imshow(sample['image'][i], cmap='gray')
    #     axarr[1, i].imshow(sample['mask'][i], cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()


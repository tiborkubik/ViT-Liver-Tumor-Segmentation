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

import glob
import logging
import os
import random

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T

from src.trainer import config
from src.trainer.transforms import Invert, RandomElastic

matplotlib.rcParams["figure.dpi"] = 400

EXT = '*.png'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class LiverTumorDataset(Dataset):

    def __init__(self, dataset_path, transforms_img=None, transforms_mask=None, training_mode='2D'):

        self.dataset_path = dataset_path
        self.transforms_img = transforms_img
        self.transforms_mask = transforms_mask
        self.training_mode = training_mode

        self.current_slice_idx = 0  # Current position of iterable
        self.slices = []  # One sample: tuple (slice .png path, mask .png path).

        w_path = os.path.join(self.dataset_path, 'vols-2d/')

        all_slice_files = glob.glob(os.path.join(glob.escape(w_path), EXT))

        if len(all_slice_files) == 0:
            logging.warning(F"No volume file found in {w_path}")

        for slice_path in all_slice_files:
            mask_path = slice_path.replace('volume', 'segmentation').replace('vols', 'segs')
            self.slices.append((slice_path, mask_path))

    def __getitem__(self, item):
        volume_path = self.slices[item][0]
        segmentation_path = self.slices[item][1]
        vol_idx = self._get_vol_idx(volume_path)

        slice_of_interest = cv2.imread(volume_path, cv2.IMREAD_GRAYSCALE)
        slice_of_interest = cv2.resize(slice_of_interest,
                                       (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                                       interpolation=cv2.INTER_AREA)

        slice_of_interest = normalize_slice(slice_of_interest)

        if self.training_mode == '2D':
            images = slice_of_interest

        elif self.training_mode == '2.5D':
            neighbors_lower_z = []
            neighbors_higher_z = []

            filename, extension = os.path.splitext(self.slices[item][0])
            volumes_prefix, volume_details = filename.split('volume')
            slice_num = int(volume_details.split('-')[-1])
            vol_num = int(volume_details.split('-')[-2])
            vol_name = volumes_prefix + 'volume'
            # Check if there are any neighbouring slices with lower z-value. Repeat slice of interest otherwise.
            if slice_num < 4:
                for i in range(0, 4):
                    neighbors_lower_z.append(slice_of_interest)
            else:
                for i in range(4, 0, -1):
                    slice = cv2.imread(vol_name + '-' + str(vol_num) + '-' + str(slice_num - i) + extension,
                                       cv2.IMREAD_GRAYSCALE)
                    slice = cv2.resize(slice, (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                                       interpolation=cv2.INTER_AREA)

                    neighbors_lower_z.append(normalize_slice(slice))

            # Check if there are any neighboring slices with higher z-value. Repeat slice of interest otherwise.
            for i in range(1, 5):
                if os.path.exists(vol_name + '-' + str(vol_num) + '-' + str(slice_num + i) + extension):
                    slice = cv2.imread(vol_name + '-' + str(vol_num) + '-' + str(slice_num + i) + extension,
                                       cv2.IMREAD_GRAYSCALE)
                    slice = cv2.resize(slice, (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                                       interpolation=cv2.INTER_AREA)

                    neighbors_higher_z.append(normalize_slice(slice))
                else:
                    neighbors_higher_z = []
                    for _ in range(1, 5):
                        neighbors_higher_z.append(slice_of_interest)

                    break

            images = neighbors_lower_z + [slice_of_interest] + neighbors_higher_z
            ...
        else:
            # Should not come to this point...
            images = ...

        mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (config.DIMENSIONS['output_net'], config.DIMENSIONS['output_net']),
                          interpolation=cv2.INTER_AREA)

        sample = {
            'images': images,
            'masks_liver': (mask == 1.0).astype(float),
            'masks_tumor': (mask == 2.0).astype(float),
        }

        seed = np.random.randint(0, 2 ** 31)
        if self.transforms_img is not None:
            for key in sample:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                if 'mask' in key:
                    sample[key] = self.transforms_mask(sample[key])
                else:
                    if self.training_mode == '2D':
                        sample[key] = self.transforms_img(sample[key])
                    else:
                        for i in range(len(sample[key])):
                            sample[key][i] = self.transforms_img(sample[key][i])

                        sample[key] = torch.stack(sample[key]).squeeze()

        # Debugging purposes
        # assert torch.all(sample['images'] == sample['masks'])

        # Visualisation purposes
        # augmentation_diff(slice, sample['images'].squeeze(), mask, sample['masks_liver'].squeeze())
        for key in sample:
            assert sample[key] is not None, \
                f'Invalid {key} in sample {self.slices[item]}'

        for key in sample:
            if isinstance(sample[key], list):
                sample[key] = np.array(sample[key])
            if isinstance(sample[key], np.ndarray):
                sample[key] = torch.from_numpy(sample[key])


        sample['masks'] = torch.concat([sample['masks_liver'], sample['masks_tumor']])
        sample['vol_idx'] = vol_idx
        del sample['masks_liver']
        del sample['masks_tumor']

        return sample

    def _get_vol_idx(self, path: str):
        vol_idx = path.split('-')[-2]
        return int(vol_idx)

    def __len__(self):
        return len(self.slices)

    def __iter__(self):
        self.current_slice_idx = 0
        return self

    def __next__(self):
        self.current_slice_idx += 1
        if self.current_slice_idx < len(self):
            return self.__getitem__(self.current_slice_idx)
        raise StopIteration


def get_dataset_loader(dataset_path, transforms_img=None, transforms_mask=None, training_mode='2D', batch_size=32,
                       workers=0, random=True, ddp=False):
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

    :param dataset_path: Path to the dataset
    :param transforms_img: Transforms applied on training images.
    :param transforms_mask: Transforms applied only on the masks.
    :param batch_size: Training batch size.
    :param workers: Number of workers for parallel data loading. Depends on the number of CPUs available.
    :param random: Randomize choice of train/val split.
    :param ddp: Enable when multi-gpu training.

    :return: Train and validation loader.
    """

    ds = LiverTumorDataset(dataset_path=dataset_path,
                           transforms_img=transforms_img,
                           transforms_mask=transforms_mask,
                           training_mode=training_mode)

    ds_len = len(ds)

    logging.info(f'Length of dataset: {ds_len}.')

    sampler = DistributedSampler(ds) if ddp else None

    loader_kwargs = {'batch_size': batch_size, 'num_workers': workers, 'pin_memory': True, 'drop_last': True}
    loader = DataLoader(ds, sampler=sampler, **loader_kwargs)

    if batch_size is not None:
        if len(loader.dataset.slices) < batch_size:
            logging.warning(f'Training data subset too small: {len(loader.dataset.indices)}.')

    return loader


def normalize_slice(ct_slice, window=(config.TYPICAL_LIVER_WW, config.TYPICAL_LIVER_WL)):
    """
    Normalize CT slice, by windowing over predefined levels. Default window param corresponds to liver window form from
    https://docs.fast.ai/medical.imaging.
    """
    px = ct_slice.copy()
    w_width, w_level = window
    px_min, px_max = w_level - w_width // 2, w_level + w_width // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max

    return (px - px_min) / (px_max - px_min)


if __name__ == '__main__':
    """ So this method is basically ready-to-use in training pipeline. Here, it's just for testing. """
    test_batch_size = 8

    transforms = [
        T.RandomApply([RandomElastic(alpha=0.5, sigma=0.05)], p=0.85),
        T.ToTensor(),
        T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=5.39)], p=0.44),
        T.RandomApply([T.RandomRotation(degrees=3.09)], p=0.59),
        T.RandomApply([T.RandomAffine(degrees=0, shear=3.68)], p=0.62),
        T.RandomApply([T.ColorJitter(brightness=0.959)], p=0.71),
        T.RandomApply([Invert()], p=0.5)
    ]

    # train_loader = get_dataset_loader('data/train/', batch_size=test_batch_size,
    #                                                transforms_img=T.Compose(transforms),
    #                                                transforms_mask=T.Compose(transforms[:-2]))

    test_loader = get_dataset_loader('dataset/test/', batch_size=test_batch_size)
    sample = next(iter(test_loader))
    pass
    #
    # f, axarr = plt.subplots(2, test_batch_size)
    # f.set_size_inches(10, 3)
    # for i in range(test_batch_size):
    #     axarr[0, i].imshow(sample['image'][i], cmap='gray')
    #     axarr[1, i].imshow(sample['mask'][i], cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()

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
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T

import src.trainer.config as config
from src.trainer.transforms import Invert, RandomElastic

matplotlib.rcParams["figure.dpi"] = 400

EXT = '*.png'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


class LiverTumorDataset(Dataset):

    def __init__(self, dataset_path, transforms_img=None, transforms_mask=None):

        self.dataset_path = dataset_path
        self.transforms_img = transforms_img
        self.transforms_mask = transforms_mask

        self.current_slice_idx = 0  # Current position of iterable
        self.slices = []  # One sample: tuple (slice .png path, mask .png path).

        w_path = os.path.join(self.dataset_path, 'vols-2d/')

        all_slice_files = [file
                           for path, subdir, files in os.walk(w_path)
                           for file in glob.glob(os.path.join(path, EXT))]

        if len(all_slice_files) == 0:
            logging.warning(F"No volume file found in {w_path}")

        for slice_path in all_slice_files:
            mask_path = slice_path.replace('volume', 'segmentation').replace('vols', 'segs')
            self.slices.append((slice_path, mask_path))

        self.slices = self.slices[:2000]

    def __getitem__(self, item):
        volume_path = self.slices[item][0]
        segmentation_path = self.slices[item][1]
        vol_idx = self._get_vol_idx(volume_path)

        slice = cv2.imread(volume_path, cv2.IMREAD_GRAYSCALE)
        slice = cv2.resize(slice, (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                           interpolation=cv2.INTER_AREA)

        mask = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (config.DIMENSIONS['output_net'], config.DIMENSIONS['output_net']),
                          interpolation=cv2.INTER_AREA)

        sample = {
            'images': torch.tensor(normalize_slice(slice)),
            'masks_liver': (mask == 1.0).astype(float),
            'masks_tumor': (mask == 2.0).astype(float),
        }

        seed = np.random.randint(0, 2 ** 32)
        if self.transforms_img is not None:
            for key in sample:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                if 'mask' in key:
                    sample[key] = self.transforms_mask(sample[key])
                else:
                    sample[key] = self.transforms_img(sample[key])

        # Debugging purposes
        # assert torch.all(sample['images'] == sample['masks'])

        # Visualisation purposes
        # augmentation_diff(slice, sample['images'].squeeze(), mask, sample['masks_liver'].squeeze())

        for key in sample:
            assert sample[key] is not None, \
                f'Invalid {key} in sample {self.slices[item]}'

        sample['masks'] = torch.concat([torch.from_numpy(sample['masks_liver']),
                                        torch.from_numpy(sample['masks_tumor'])])
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


def get_dataset_loader(dataset_path, transforms_img=None, transforms_mask=None, batch_size=32, workers=0,
                       random=True, ddp=False):
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

    :param train_path: Path to the dataset of TRAINING part only!
    :param val_path: Path to the dataset of VALIDATION part only!
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
                           transforms_mask=transforms_mask)

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


def pre_process_niis(path):
    """
    Use this to extract slices and corresponding masks from .nii volumes as png images and store them on disk.

    :param path: Path to the root of dataset folder.

    :return: void
    """
    w_path = os.path.join(path, 'vols-3d')
    vol_slices_path = os.path.join(path, 'vols-2d')
    segs_slices_path = os.path.join(path, 'segs-2d')

    if not os.path.exists(vol_slices_path):
        os.makedirs(vol_slices_path)
    if not os.path.exists(segs_slices_path):
        os.makedirs(segs_slices_path)

    # Paths of .nii volumes.
    all_nii_files = [file
                     for path, subdir, files in os.walk(w_path)
                     for file in glob.glob(os.path.join(path, '*.nii'))]

    if len(all_nii_files) == 0:
        logging.warning(F"No .nii file was found in {path}")

    for i, nii_file in enumerate(all_nii_files):
        print(f'[{i}/{len(all_nii_files)}] Processing file {nii_file}.')

        vol = nib.load(nii_file)
        vol = vol.get_fdata()
        num_slices = vol.shape[2]  # The shape is denoted e.g (512, 512, 826), where the last one is num of slices.

        vol_name = nii_file.split(".")[-2].split('/')[-1]

        for i in range(num_slices):
            cv2.imwrite(os.path.join(path, f'vols-2d/{vol_name}-{i}.png'), vol[:, :, i])

        seg_nii_path = nii_file.replace('volume', 'segmentation').replace('vols', 'segs')
        seg = nib.load(seg_nii_path)
        seg = seg.get_fdata()

        seg_name = seg_nii_path.split(".")[-2].split('/')[-1]

        """ Uncomment this to plot slice, mask and normalized slice sample"""
        # if nii_file == 'data/train-val/vols-3d/volume-2.nii':
        #     sample_id = 450
        #     plot_slice_sample(vol[..., sample_id], normalize_slice(vol[..., sample_id].astype(np.float32)), seg[..., sample_id], savefig='documentation/slice_sample')
        for i in range(num_slices):
            cv2.imwrite(os.path.join(path, f'segs-2d/{seg_name}-{i}.png'), seg[:, :, i])


if __name__ == '__main__':
    """ Uncomment this to extract the individual slices and masks as png images (from nii volumes). """
    # pre_process_niis('dataset/test/')

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

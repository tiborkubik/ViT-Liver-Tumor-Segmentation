import argparse
import os
from typing import List, Tuple

import cv2
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.metrics.DicePerVolume import DicePerVolume, VolumeMetric
from src.networks.utils import create_model
from src.trainer import config
from src.trainer.LiverTumorDataset import LiverTumorDataset, normalize_slice


class Evaluator:

    def __init__(self, dataset_path, model, device, liver_metrics: List[VolumeMetric],
                 lesion_metrics: List[VolumeMetric]):
        self.dataset_path = dataset_path
        self.model = model
        self.device = device
        self.liver_metrics = liver_metrics
        self.lesion_metrics = lesion_metrics
        self.dataset = LiverTumorDataset(dataset_path=dataset_path)

    def evaluate(self, volumes=None):
        if volumes is None:
            volumes = len(self.dataset)

        self.reset_metrics(self.liver_metrics)
        self.reset_metrics(self.lesion_metrics)

        self.model.eval()
        loop = tqdm(self.dataset, total=volumes, desc="Evaluation")

        with torch.no_grad():
            for i_batch, sample in enumerate(loop):
                inputs = sample['images'].type(torch.FloatTensor).to(self.device)
                masks = sample['masks'].type(torch.FloatTensor).to(self.device)
                vol_idx = sample['vol_idx']

                # Add batch and channel dimension
                inputs_batch = inputs.unsqueeze(0).unsqueeze_(0)
                # Forward
                predictions = self.model(inputs_batch)
                masks_reshaped = torch.reshape(masks, predictions.size())

                preds_liver = predictions[:, 0, :, :]
                preds_lesion = predictions[:, 0, :, :]

                masks_liver = masks_reshaped[:, 0, :, :]
                masks_lesion = masks_reshaped[:, 1, :, :]

                for metric in self.liver_metrics:
                    metric.update(preds_liver, masks_liver, vol_idx)

                for metric in self.lesion_metrics:
                    metric.update(preds_lesion, masks_lesion, vol_idx)

    def reset_metrics(self, metrics):
        for metric in metrics:
            metric.reset()

    def create_nii(self, volume_idx: int, save_path: str) -> None:
        volume_path = os.path.join(self.dataset_path, 'vols-3d', F"volume-{volume_idx}.nii")
        volume_image = nib.load(volume_path)
        volume_data = volume_image.get_fdata()
        num_slices = volume_data.shape[2]

        liver_masks = []
        tumor_masks = []

        with torch.no_grad():
            for slice_idx in range(num_slices):
                slice = self._prepare_slice(volume_data, slice_idx)
                inputs = torch.tensor(slice).type(torch.FloatTensor).to(self.device)

                # Add batch and channel dimension
                inputs_batch = inputs.unsqueeze(0).unsqueeze_(0)

                slice_predictions = self.model(inputs_batch)[0]

                liver_mask, tumor_mask = self._postprocesses_mask(slice_predictions.numpy(),
                                                                  new_size=volume_data.shape[:2],
                                                                  threshold=0.3)
                liver_masks.append(liver_mask)
                tumor_masks.append(tumor_mask)

        # Stack all slices into a single array
        liver_slices = np.stack(liver_masks, -1)
        tumor_slices = np.stack(tumor_masks, -1)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save as .nii files
        liver_mask_path = os.path.join(save_path, f"segmentation_liver_{volume_idx}.nii")
        tumor_mask_path = os.path.join(save_path, f"segmentation_tumor_{volume_idx}.nii")
        nib.save(nib.Nifti1Image(liver_slices, affine=volume_image.affine), liver_mask_path)
        nib.save(nib.Nifti1Image(tumor_slices, affine=volume_image.affine), tumor_mask_path)

    def _prepare_slice(self, volume, slice_idx):
        slice = volume[:, :, slice_idx]
        resized_slice = cv2.resize(slice, (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                                   interpolation=cv2.INTER_AREA)
        normalized_slice = normalize_slice(resized_slice)
        return normalized_slice

    def _postprocesses_mask(self, pred_mask: np.ndarray, new_size: Tuple, threshold: float) -> Tuple[
        np.ndarray, np.ndarray]:
        binarized_mask = (pred_mask > threshold).astype(np.float32)
        liver_mask = binarized_mask[0]
        tumor_mask = binarized_mask[1]
        resized_liver_mask = cv2.resize(liver_mask, new_size, interpolation=cv2.INTER_AREA)
        resized_tumor_mask = cv2.resize(tumor_mask, new_size, interpolation=cv2.INTER_AREA)
        return resized_liver_mask, resized_tumor_mask


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Liver and Liver Tumor Segmentation from CT Scans of '
                                                 'Human Abdomens',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', metavar='D', type=str,
                        default=None, help='Path to 2D slices')
    parser.add_argument('-w', '--weights', metavar='W', type=str,
                        default='trained_weights/UNet/03-25-18-49-14-UNet.pt', help='Trained model weights')
    parser.add_argument('-n', '--network-name', metavar='NN', type=str,
                        default='UNet', help='Network name')
    args = parser.parse_args()

    assert args.dataset is not None
    assert args.weights is not None
    assert args.network_name in ['UNet', 'AttentionUNet', 'TransUNet']

    return args


if __name__ == "__main__":
    args = parse_args()
    model = create_model(args.network_name, args.weights)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    liver_metrics = [DicePerVolume()]
    lesion_metrics = [DicePerVolume()]
    evaluator = Evaluator(args.dataset, model, device, liver_metrics, lesion_metrics)

    evaluator.evaluate()
    print('Liver')
    print('Per volume dice score:', liver_metrics[0].compute_per_volume())
    print('Dice score:', liver_metrics[0].compute_total())
    print('Lesion')
    print('Per volume dice score:', lesion_metrics[0].compute_per_volume())
    print('Dice score:', lesion_metrics[0].compute_total())

    # evaluator.create_nii(0, 'dataset/predictions')

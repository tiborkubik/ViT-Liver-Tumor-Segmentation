import argparse
import os
import nibabel as nib
import logging
import numpy as np
import zipfile
import glob
import cv2
import torch

from tqdm import tqdm
from src.trainer import config
from typing import List, Tuple
from src.evaluation.metrics import ASSD, DicePerVolume, MSD, RAVD, VOE, VolumeMetric
from src.evaluation.utils import write_metrics
from src.networks.utils import create_model
from src.trainer.LiverTumorDataset import LiverTumorDataset, normalize_slice

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class Evaluator:

    def __init__(self, dataset_path, model, device, liver_metrics: List[VolumeMetric],
                 lesion_metrics: List[VolumeMetric], apply_masking: bool,
                 apply_morphological: bool, kernel_liver: int, kernel_tumor: int, training_mode='2D'):
        self.dataset_path = dataset_path
        self.model = model
        self.device = device
        self.liver_metrics = liver_metrics
        self.lesion_metrics = lesion_metrics
        self.dataset = LiverTumorDataset(dataset_path=dataset_path, training_mode=training_mode)

        self.apply_masking = apply_masking
        self.apply_morphological = apply_morphological
        self.kernel_liver = kernel_liver
        self.kernel_tumor = kernel_tumor

    def evaluate(self, volumes=None):
        if volumes is None:
            volumes = len(self.dataset)

        self._reset_metrics(self.liver_metrics)
        self._reset_metrics(self.lesion_metrics)

        self.model.eval()
        loop = tqdm(self.dataset, total=volumes, desc="Evaluation")

        with torch.no_grad():
            for i_batch, sample in enumerate(loop):
                inputs, masks, vol_idx = self._prepare_sample(sample)

                predictions = self.model(inputs)

                preds_liver = predictions[:, 0, :, :]
                preds_lesion = predictions[:, 1, :, :]

                masks_reshaped = torch.reshape(masks, predictions.size())
                masks_liver = masks_reshaped[:, 0, :, :]
                masks_lesion = masks_reshaped[:, 1, :, :]

                preds_liver = preds_liver.cpu()
                preds_lesion = preds_lesion.cpu()
                masks_liver = masks_liver.cpu()
                masks_lesion = masks_lesion.cpu()

                preds_liver_np = preds_liver.numpy()
                preds_liver_np = (preds_liver_np > 0.3).astype(np.float32)

                preds_lesion_np = preds_lesion.numpy()
                preds_lesion_np = (preds_lesion_np > 0.3).astype(np.float32)

                preds_liver_np, preds_lesion_np = self.postprocess(preds_liver_np,
                                                                   preds_lesion_np)

                preds_liver = torch.from_numpy(preds_liver_np)
                preds_lesion = torch.from_numpy(preds_lesion_np)

                self._update_metrics(self.liver_metrics, preds_liver, masks_liver, vol_idx)
                self._update_metrics(self.lesion_metrics, preds_lesion, masks_lesion, vol_idx)

    def _prepare_sample(self, sample):
        inputs = sample['images'].type(torch.FloatTensor).to(self.device)
        masks = sample['masks'].type(torch.FloatTensor).to(self.device)
        vol_idx = sample['vol_idx']
        # Add batch and channel dimension
        inputs_batch = inputs.unsqueeze(0).unsqueeze_(0)
        return inputs_batch, masks, vol_idx

    @staticmethod
    def _reset_metrics(metrics):
        for metric in metrics:
            metric.reset()

    @staticmethod
    def _update_metrics(metrics, preds, masks, vol_idx):
        for metric in metrics:
            metric.update(preds, masks, vol_idx)

    @staticmethod
    def _combine_liver_and_tumor_slices(liver_slices, tumor_slices):
        # Tumor should be indicated by value 2
        tumor_slices *= 2

        # Combine volumes together
        slices = liver_slices + tumor_slices
        # Replace value 3 (liver + lesion) with 2 (lesion)
        slices[slices > 2] = 2
        return slices

    def _postprocesses_mask(self, pred_mask: np.ndarray, new_size: Tuple, threshold: float) -> Tuple[
        np.ndarray, np.ndarray]:
        binarized_mask = (pred_mask > threshold).astype(np.float32)
        liver_mask = binarized_mask[0]
        tumor_mask = binarized_mask[1]
        resized_liver_mask = cv2.resize(liver_mask, new_size, interpolation=cv2.INTER_AREA)
        resized_tumor_mask = cv2.resize(tumor_mask, new_size, interpolation=cv2.INTER_AREA)

        liver_mask_postprocessed, tumor_mask_postprocessed = self.postprocess(resized_liver_mask, resized_tumor_mask)

        return liver_mask_postprocessed, tumor_mask_postprocessed

    def postprocess(self, resized_liver_mask, resized_tumor_mask):
        if self.apply_morphological:
            kernel_liver = np.ones((self.kernel_liver, self.kernel_liver), np.uint8)
            kernel_tumor = np.ones((self.kernel_tumor, self.kernel_tumor), np.uint8)

            # Apply morphological operations on liver prediction.
            liver_mask_postprocessed = cv2.morphologyEx(resized_liver_mask, cv2.MORPH_OPEN, kernel_liver)
            liver_mask_postprocessed = cv2.morphologyEx(liver_mask_postprocessed, cv2.MORPH_CLOSE, kernel_liver)

            # Apply morphological operations on tumor prediction.
            tumor_mask_postprocessed = cv2.morphologyEx(resized_tumor_mask, cv2.MORPH_OPEN, kernel_tumor)
            tumor_mask_postprocessed = cv2.morphologyEx(tumor_mask_postprocessed, cv2.MORPH_CLOSE, kernel_tumor)
        else:
            liver_mask_postprocessed = resized_liver_mask
            tumor_mask_postprocessed = resized_tumor_mask

        if self.apply_masking:
            tumor_mask_postprocessed = liver_mask_postprocessed.astype(bool) & tumor_mask_postprocessed.astype(bool)
            tumor_mask_postprocessed = tumor_mask_postprocessed.astype(np.float32)

        return liver_mask_postprocessed, tumor_mask_postprocessed

    def generate_zip(self, save_dir: str, zip_name: str) -> None:
        os.makedirs(args.zip_location, exist_ok=True)
        vols_pattern = os.path.join(glob.escape(self.dataset_path), 'vols-3d', '*')
        vol_file_paths = glob.glob(vols_pattern)
        for volume_idx, volume_path in enumerate(vol_file_paths):
            self.create_nii(volume_path, save_dir, volume_idx)
        self._save_files_to_zip(save_dir, zip_name)

    @staticmethod
    def _save_files_to_zip(dir: str, zip_name: str):
        vols_pattern = os.path.join(glob.escape(dir), '*.nii')
        vol_files = glob.glob(vols_pattern)
        zip_path = os.path.join(dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for vol_filepath in vol_files:
                vol_filename = os.path.basename(vol_filepath)
                zipf.write(vol_filepath, vol_filename)

    def create_nii(self, volume_path: str, save_path: str, volume_idx: int) -> None:
        volume_image = nib.load(volume_path)
        volume_data = volume_image.get_fdata()
        num_slices = volume_data.shape[2]
        logging.debug(F"Volume shape: {volume_data.shape}")

        liver_masks = []
        tumor_masks = []

        with torch.no_grad():
            for slice_idx in range(num_slices):
                slice = self._prepare_slice(volume_data, slice_idx)
                inputs = torch.tensor(slice).type(torch.FloatTensor).to(self.device)

                # Add batch and channel dimension
                inputs_batch = inputs.unsqueeze(0).unsqueeze_(0)

                slice_predictions = self.model(inputs_batch)[0]

                liver_mask, tumor_mask = self._postprocesses_mask(slice_predictions.cpu().numpy(),
                                                                  new_size=volume_data.shape[:2],
                                                                  threshold=0.3)
                liver_masks.append(liver_mask)
                tumor_masks.append(tumor_mask)

        # Stack all slices into a single array
        liver_slices = np.stack(liver_masks, -1)
        tumor_slices = np.stack(tumor_masks, -1)
        slices = self._combine_liver_and_tumor_slices(liver_slices, tumor_slices)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logging.debug(F"Created segmentation volume with shape: {slices.shape}")
        # Save as .nii files
        segmentation_path = os.path.join(save_path, f"test-segmentation-{volume_idx}.nii")
        nib.save(nib.Nifti1Image(slices, affine=volume_image.affine), segmentation_path)

    @staticmethod
    def _prepare_slice(volume, slice_idx):
        slice = volume[:, :, slice_idx]
        resized_slice = cv2.resize(slice, (config.DIMENSIONS['input_net'], config.DIMENSIONS['input_net']),
                                   interpolation=cv2.INTER_AREA)
        normalized_slice = normalize_slice(resized_slice)
        return normalized_slice


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Liver and Liver Tumor Segmentation from CT Scans of '
                                                 'Human Abdomens',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', metavar='D', type=str,
                        default=None, help='Path to 2D slices', required=True)
    parser.add_argument('--zip-location', type=str,
                        default='dataset/segmentations', help='Folder location of the output zip file')
    parser.add_argument('-w', '--weights', metavar='W', type=str,
                        default='trained_weights/UNet/03-25-18-49-14-UNet.pt', help='Trained model weights')
    parser.add_argument('-n', '--network-name', metavar='NN', type=str,
                        default='UNet', help='Network name')
    parser.add_argument('-p1', '--postprocess-masking', metavar='P1', action='store_true', dest='apply_masking',
                        help='Apply a postprocessing, where the tumor parts detected out of the liver mask is not'
                             'considered.')
    parser.add_argument('-p2', '--postprocess-morphological', metavar='P2', action='store_true',
                        dest='apply_morphological', help='Apply morphological operations on detected masks to get rid'
                                                         'of holes and noise.')
    parser.add_argument('-kl', '--kernel-liver', metavar='KL', type=int, default=15,
                        dest='kernel_liver', help='Size of kernel for morphological post-processing on liver.')
    parser.add_argument('-kt', '--kernel-tumor', metavar='KT', type=int, default=3,
                        dest='kernel_tumor', help='Size of kernel for morphological post-processing on tumor.')

    parser.add_argument('-sp', '--save-prefix', metavar='SP', type=str,
                        default="", help='Prefix of path to save outputs',
                        dest='save_prefix')
    parser.add_argument('-z', '--generate-zip', action='store_true', dest='generate_zip',
                        help='Whether to generate zip. Performs evaluation without creating volumes instead if false.')

    args = parser.parse_args()

    assert args.dataset is not None
    assert args.weights is not None
    assert args.network_name in ['UNet', 'AttentionUNet', 'TransUNet']

    return args


if __name__ == "__main__":
    args = parse_args()
    model = create_model(args.network_name, weights_path=args.weights, training_mode='2D')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    liver_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
    lesion_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
    evaluator = Evaluator(args.dataset, model, device, liver_metrics, lesion_metrics,
                          args.apply_masking, args.apply_morphological, args.kernel_liver, args.kernel_tumor)

    if args.generate_zip:
        evaluator.generate_zip(args.zip_location, 'submission.zip')
    else:
        evaluator.evaluate()
        metrics_path = os.path.join(args.save_prefix, 'metrics.log')
        write_metrics(metrics_path, 'Liver', liver_metrics)
        write_metrics(metrics_path, 'Lesion', lesion_metrics)

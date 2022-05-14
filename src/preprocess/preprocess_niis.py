import glob
import logging
import os

import cv2
import nibabel as nib
import argparse

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


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
    all_nii_files = glob.glob(os.path.join(w_path, '*.nii'))

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
    parser = argparse.ArgumentParser(description='Dataset preprocessor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dp', '--dataset-path', metavar='D', type=str,
                        default=None, help='Path to 3d vols and segs', dest='dataset_path')

    args = parser.parse_args()

    pre_process_niis(args.dataset_path)

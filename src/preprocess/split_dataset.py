import logging
import os

import argparse

import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
EXT = ".png"


def split_dataset(path, valid_split):
    volumes_path = os.path.join(path, 'vols-3d/')
    volumes = np.array(os.listdir(volumes_path))
    n_volumes = len(volumes)
    validation_mask = np.zeros(n_volumes, dtype=bool)
    validation_part = int(n_volumes * valid_split)
    validation_indexes = np.random.choice(n_volumes, validation_part, replace=False)
    validation_mask[validation_indexes] = True

    train_split, val_split = (volumes[~validation_mask], 'train'), (volumes[validation_mask], 'val')

    for split, split_name in (train_split, val_split):
        split_path = os.path.join(path, split_name)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        for part in ['segs-3d', 'vols-3d']:
            part_path = os.path.join(split_path, part)
            if not os.path.exists(part_path):
                os.makedirs(part_path)
            orig_path = os.path.join(path, part)
            for volume in split:
                if 'seg' in part:
                    volume = volume.replace('volume', 'segmentation')
                os.replace(os.path.join(orig_path, volume), os.path.join(part_path, volume))
    for part in ['segs-3d', 'vols-3d']:
        orig_path = os.path.join(path, part)
        assert len(os.listdir(orig_path)) == 0
        os.rmdir(orig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset preprocessor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dp', '--dataset-path', metavar='D', type=str,
                        default=None, help='Path to 2d vols and segs', dest='dataset_path')
    parser.add_argument('-v', '--validation_part', metavar='V', type=float,
                        default=0.15, help='Percentage of validation split', dest='val_split')

    args = parser.parse_args()

    split_dataset(args.dataset_path, args.val_split)

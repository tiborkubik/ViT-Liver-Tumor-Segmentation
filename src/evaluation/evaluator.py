import argparse
from typing import List

import torch
from tqdm import tqdm

from src.evaluation.metrics.DicePerVolume import DicePerVolume, VolumeMetric
from src.networks.utils import create_model
from src.trainer.LiverTumorDataset import LiverTumorDataset


class Evaluator:

    def __init__(self, dataset_path, model, device, metrics: List[VolumeMetric]):
        self.dataset_path = dataset_path
        self.model = model
        self.device = device
        self.metrics = metrics
        self.dataset = LiverTumorDataset(dataset_path=dataset_path)

    def evaluate(self, volumes=None):
        if volumes is None:
            volumes = len(self.dataset)

        for metric in self.metrics:
            metric.reset()

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

                for metric in self.metrics:
                    metric.update(predictions, masks_reshaped, vol_idx)

    def create_nii(self, volume_idx):
        # For each slice
        # Predict liver mask
        # Predict tumor mask
        # Append both
        # Save as nii
        pass


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
    dice_metric = DicePerVolume()
    metrics = [dice_metric]
    evaluator = Evaluator(args.dataset, model, device, metrics)
    evaluator.evaluate()

    print('Per volume dice score:', dice_metric.compute_per_volume())
    print('Dice score:', dice_metric.compute_total())

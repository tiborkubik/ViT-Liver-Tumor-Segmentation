import argparse
import os
import logging

import torch
from src.evaluation.metrics import ASSD, DicePerVolume, MSD, RAVD, VOE
from src.evaluation.utils import write_metrics
from src.networks.utils import create_model
from src.evaluation.evaluator import Evaluator
from src.trainer import config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Liver and Liver Tumor Segmentation from CT Scans of '
                                                 'Human Abdomens',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', metavar='D', type=str,
                        default=None, help='Path to 2D slices', required=True)
    parser.add_argument('-w', '--weights', metavar='W', type=str,
                        default='trained_weights/UNet/03-25-18-49-14-UNet.pt', help='Trained model weights')
    parser.add_argument('-n', '--network-name', metavar='NN', type=str,
                        default='UNet', help='Network name')
    parser.add_argument('-sp', '--save-prefix', metavar='SP', type=str,
                        default="", help='Prefix of path to save outputs',
                        dest='save_prefix')
    parser.add_argument('-tm', '--training-mode', metavar='TM', type=str,
                        default=config.HYPERPARAMETERS['training_mode'], help='2D or 2.5D training...',
                        dest='training_mode')
    args = parser.parse_args()

    assert args.dataset is not None
    assert args.weights is not None
    assert args.network_name in ['UNet', 'AttentionUNet', 'TransUNet']

    return args


def evaluate(model, device, dataset, metrics_path, training_mode):
    for apply_morphological in [True, False]:
        if not apply_morphological:
            for apply_masking in [True, False]:
                setting = f'liver_kernel: 0, tumor_kernel: 0, ' \
                          f'apply_morphological: {apply_morphological}, apply_masking: {apply_masking}'
                logging.debug(setting)
                liver_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
                lesion_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
                evaluator = Evaluator(dataset, model, device, liver_metrics, lesion_metrics,
                                      apply_masking, apply_morphological, 0, 0, training_mode)
                evaluator.evaluate()
                write_metrics(metrics_path, f'Liver {setting}', liver_metrics)
                write_metrics(metrics_path, f'Lesion {setting}', lesion_metrics)
        else:
            for apply_masking in [True, False]:
                for l_kernel in [1, 2, 4] if apply_morphological else [None]:
                    for t_kernel in [1, 3] if apply_morphological else [None]:
                        liver_kernel = 2 ** l_kernel
                        tumor_kernel = 2 ** t_kernel
                        setting = f'liver_kernel: {liver_kernel}, tumor_kernel: {tumor_kernel}, ' \
                                  f'apply_morphological: {apply_morphological}, apply_masking: {apply_masking}'
                        logging.debug(setting)
                        liver_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
                        lesion_metrics = [DicePerVolume(), VOE(), RAVD(), ASSD(), MSD()]
                        evaluator = Evaluator(dataset, model, device, liver_metrics, lesion_metrics,
                                              apply_masking, apply_morphological, liver_kernel, tumor_kernel,
                                              training_mode)
                        evaluator.evaluate()
                        write_metrics(metrics_path, f'Liver {setting}', liver_metrics)
                        write_metrics(metrics_path, f'Lesion {setting}', lesion_metrics)


if __name__ == "__main__":
    args = parse_args()
    model = create_model(args.network_name, weights_path=args.weights, training_mode=args.training_mode)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    metrics_path = os.path.join(args.save_prefix, 'metrics.log')
    evaluate(model, device, args.dataset, metrics_path, args.training_mode)

"""
    :filename train.py

    :brief Script containing the entry point for the training.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""
import torch
import config
import logging
import argparse

from src.networks.UNet import UNet
from src.trainer.Trainer import Trainer
from src.networks.AttentionUNet import AttentionUNet
# from src.networks.TransUNet import TransUNet


def parse_args():
    """Argument parsing for the config overwrite."""
    parser = argparse.ArgumentParser(description='Liver and Liver Tumor Segmentation from CT Scans of Human Abdomens',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dt', '--dataset-train', metavar='D', type=str,
                        default=None, help='Path to 2D slices w. train slices.', dest='dataset_train')
    parser.add_argument('-dv', '--dataset-val', metavar='D', type=str,
                        default=None, help='Path to 2D slices w. val slices.', dest='dataset_val')
    parser.add_argument('-e', '--epochs', metavar='E', type=int,
                        default=config.HYPERPARAMETERS['epochs'], help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?',
                        default=config.HYPERPARAMETERS['batch_size'], help='Batch size', dest='batch_size')
    parser.add_argument('-lo', '--loss', metavar='LO', type=str,
                        default=config.HYPERPARAMETERS['loss'], help='Loss function for training', dest='loss')
    parser.add_argument('-w', '--weight-decay', metavar='WD', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['weight_decay'], help='Weight decay', dest='weight_decay')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['learning_rate'], help='Learning rate', dest='lr')
    parser.add_argument('-bt', '--betas', metavar='BT', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['betas'], help='Coefficients used for computing running averages of gradient and its square.', dest='betas')
    parser.add_argument('-p', '--eps', metavar='EP', type=float, nargs='?',
                        default=config.HYPERPARAMETERS['adam_w_eps'], help='Term added to the denominator to improve numerical stability.', dest='adam_w_eps')
    parser.add_argument('-n', '--network-name', metavar='NN', type=str,
                        default=config.HYPERPARAMETERS['network'], help='Network name', dest='network_name')
    parser.add_argument('-s', '--early-stopping', metavar='ES', type=bool,
                        default=True, help='Apply early stopping', dest='early_stopping')
    parser.add_argument('-lp', '--lr-patience', metavar='LP', type=float,
                        default=config.HYPERPARAMETERS['lr_scheduler_patience'], help='Lr patience', dest='lr_scheduler_patience')
    parser.add_argument('-lm', '--lr-min', metavar='LM', type=float,
                        default=config.HYPERPARAMETERS['lr_scheduler_min_lr'], help='Minimal lr value',
                        dest='lr_scheduler_min_lr')
    parser.add_argument('-lf', '--lr-factor', metavar='LF', type=float,
                        default=config.HYPERPARAMETERS['lr_scheduler_factor'], help='Factor of lr lowering',
                        dest='lr_scheduler_factor')
    parser.add_argument('-wl', '--weight-liver', metavar='WL', type=float,
                        default=config.HYPERPARAMETERS['w_liver'], help='Weight of Liver in training',
                        dest='w_liver')
    parser.add_argument('-wt', '--weight-tumor', metavar='WT', type=float,
                        default=config.HYPERPARAMETERS['w_tumor'], help='Weight of Tumor in training',
                        dest='w_tumor')
    args = parser.parse_args()

    assert args.dataset_train is not None
    assert args.dataset_val is not None
    assert args.network_name in ['UNet', 'AttentionUNet', 'TransUNet']
    assert args.loss in ['MSE', 'Dice', 'BCE', 'DiceBCE']

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()

    in_channels = 1  # BW images
    out_channels = 2  # Liver & Tumor mask

    if args.network_name == 'UNet':
        network = UNet(in_channels=in_channels, out_channels=out_channels,
                       batch_norm=True, decoder_mode='upconv')
    elif args.network_name == 'AttentionUNet':
        network = AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        ... # TODO @Lakoc
        # network = TransUNet()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network.to(device)

    logging.info(f'Device used: {device}')
    logging.info(f'GPU name: {torch.cuda.get_device_name(0)}')
    logging.info(f'Training the weights of {args.network_name}')

    try:
        trainer = Trainer(network=network,
                          network_name=args.network_name,
                          device=device,
                          dataset_train=args.dataset_train,
                          dataset_val=args.dataset_val,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          loss=args.loss,
                          weight_decay=args.weight_decay,
                          betas=args.betas,
                          adam_w_eps=args.adam_w_eps,
                          early_stopping=args.early_stopping,
                          lr=args.lr,
                          lr_scheduler_patience=args.lr_scheduler_patience,
                          lr_scheduler_min_lr=args.lr_scheduler_min_lr,
                          lr_scheduler_factor=args.lr_scheduler_factor,
                          w_liver=args.w_liver,
                          w_tumor=args.w_tumor)

        trainer.training()

    except KeyboardInterrupt:
        torch.save(network.state_dict(), 'interrupted_model.pt')

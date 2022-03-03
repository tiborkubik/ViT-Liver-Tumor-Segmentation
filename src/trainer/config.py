"""
    :filename config.py

    :brief Configs for the training part of the project.

    :author Tibor Kubik
    :author Ladislav Ondris
    :author Alexander Polok

    :email xkubik34@stud.fit.vutbr.cz
    :email xondri07@stud.fit.vutbr.cz
    :email xpolok03@stud.fit.vutbr.cz

    This file was created as a part of project called 'Visual Transformers for Liver and Liver Tumor Segmentation from
    CT Scans of Human Abdomens' for KNN/2021L course.
"""

HYPERPARAMETERS = {
    'network': 'UNet',

    'batch_size': 32,
    'epochs': 3,
    'learning_rate': 1e-3,

    'lr_scheduler_patience': 5,
    'lr_scheduler_min_lr': 1e-6,
    'lr_scheduler_factor': 0.5,

    'early_stopping_patience': 20,

    'betas': (0.9, 0.999),
    'adam_w_eps': 1e-8,
    'weight_decay': 1e-2,

    'w_liver': 1.0,
    'w_tumor': 1.0,
}
'''Hyperparameter configuration.'''

'''Input and output dimension configuration.'''
DIMENSIONS = {
    'original': 512,
    'input_net': 128,
    'output_net': 128
}

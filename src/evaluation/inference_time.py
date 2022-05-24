import glob
from typing import List

import numpy as np
import torch

from src.networks.utils import create_model


class ModelRepr:

    def __init__(self, name, weights_path, mode):
        self.name = name
        self.weights_path = weights_path
        self.mode = mode

    def __repr__(self):
        return F"Model({self.name}, {self.mode})"


def predict_single_slice(model: torch.Module):
    pass


def average_inference(model: torch.Module, reps: int):
    measured_times = []
    for i in range(reps):
        time = predict_single_slice(model)
        measured_times.append(time)
    mean_inference_time = np.mean(np.array(measured_times))
    return mean_inference_time


def parse_weights(weights: str) -> ModelRepr:

    return ModelRepr(name, weights, mode)


def get_model_representations() -> List[ModelRepr]:
    all_weights = glob.glob('trained_weights/*/*.pt')
    reprs = [parse_weights(weights) for weights in all_weights]
    return reprs


def measure_inference(model):
    model_reprs: [ModelRepr] = get_model_representations()
    for model_repr in model_reprs:
        model = create_model(model_repr.name,
                             weights_path=model_repr.weights_path,
                             training_mode=model_repr.mode)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        inference_time = average_inference(model, reps=30)
        print(F"{repr(model_repr)}: {inference_time:.2f} ms")

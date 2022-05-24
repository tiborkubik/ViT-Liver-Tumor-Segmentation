import glob
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.evaluator import prepare_sample
from src.evaluation.generate_experiment_configs import get_configs
from src.networks.utils import create_model
from src.trainer.LiverTumorDataset import get_dataset_loader


class ModelRepr:

    def __init__(self, name, weights_path, mode):
        self.name = name
        self.weights_path = weights_path
        self.mode = mode

    def __repr__(self):
        return F"Model({self.name}, {self.mode})"


def current_time_ms() -> int:
    return round(time.time() * 1000)


class Inferencer:

    def __init__(self, dataset_path: str, device):
        self.dataset_2d = get_dataset_loader(dataset_path=dataset_path,
                                             training_mode='2D',
                                             batch_size=1)
        self.dataset_25d = get_dataset_loader(dataset_path=dataset_path,
                                              training_mode='2.5D',
                                              batch_size=1)
        self.sample_2d, masks, vol_idx = prepare_sample(next(iter(self.dataset_2d)), '2D', device)
        self.sample_25d, masks, vol_idx = prepare_sample(next(iter(self.dataset_25d)), '2.5D', device)

    def predict_single_slice(self, model: torch.nn.Module, mode: str):
        with torch.no_grad():
            start_time = current_time_ms()
            if mode == '2D':
                model(self.sample_2d)
            elif mode == '2.5D':
                model(self.sample_25d)
            else:
                raise RuntimeError(F"Invalid inference mode: {mode}. "
                                   F"Valid options are: 2D, 2.5D")
            end_time = current_time_ms()
        inference_time = end_time - start_time
        return inference_time

    def average_inference(self, model: torch.nn.Module, mode: str, reps: int):
        measured_times = []
        # Warm-up, the first run should be counted
        _ = self.predict_single_slice(model, mode)
        for i in range(reps):
            time = self.predict_single_slice(model, mode)
            measured_times.append(time)
        mean_inference_time = np.mean(np.array(measured_times))
        return mean_inference_time


def parse_weights(weights: str, configs) -> ModelRepr:
    path_splits = weights.split('/')
    name = path_splits[-2]
    index = int(path_splits[-4]) - 1
    config = configs[index]
    config_splits = config.split(' ')
    mode = config_splits[2]
    # The model's name in the config should be the same as the folder containing its weights
    assert (config_splits[4] == name)
    return ModelRepr(name, weights, mode)


def get_model_representations() -> List[ModelRepr]:
    configs = get_configs()
    all_weights = glob.glob('trained_weights/*/trained-weights/*/best-weights.pt')
    reprs = [parse_weights(weights, configs) for weights in all_weights]
    return reprs


def measure_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inferencer = Inferencer('dataset/test', device)
    model_reprs: [ModelRepr] = get_model_representations()
    for model_repr in model_reprs:
        model = create_model(model_repr.name,
                             weights_path=model_repr.weights_path,
                             training_mode=model_repr.mode)
        model.to(device)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        inference_time = inferencer.average_inference(model, model_repr.mode, reps=30)
        print(F"{repr(model_repr)}: {inference_time:.0f} ms")
        print(F"Trainable params: {model_params}")


if __name__ == "__main__":
    measure_inference()

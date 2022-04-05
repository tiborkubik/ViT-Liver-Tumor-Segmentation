from abc import ABC, abstractmethod

import torch
from medpy import metric


class VolumeMetric(ABC):

    @abstractmethod
    def update(self, pred_slice: torch.Tensor, target_slice: torch.Tensor, vol_idx: int):
        pass

    @abstractmethod
    def compute_total(self):
        pass

    @abstractmethod
    def compute_per_volume(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class DicePerVolume(VolumeMetric):

    def __init__(self, pred_threshold=0.3):
        super().__init__()
        self.pred_threshold = pred_threshold
        self.vols_sum = {}
        self.vols_counts = {}

    def update(self, pred_slice: torch.Tensor, target_slice: torch.Tensor, vol_idx: int):
        if vol_idx not in self.vols_sum:
            self.vols_sum[vol_idx] = torch.tensor(0, dtype=torch.float32)
            self.vols_counts[vol_idx] = torch.tensor(0, dtype=torch.long)

        pred_slice = (pred_slice > self.pred_threshold).float()
        dice_score = metric.dc(pred_slice.numpy(), target_slice.numpy())
        self.vols_sum[vol_idx] += dice_score
        self.vols_counts[vol_idx] += 1

    def compute_total(self):
        per_volume_scores = self.compute_per_volume()
        scores = per_volume_scores.values()
        scores_sum = sum(scores)
        scores_num = len(scores)
        score = scores_sum / scores_num
        return score

    def compute_per_volume(self):
        vols_scores = {}
        for vol_idx in self.vols_sum.keys():
            vols_scores[vol_idx] = self.vols_sum[vol_idx] / self.vols_counts[vol_idx]
        return vols_scores

    def reset(self):
        self.vols_sum.clear()
        self.vols_counts.clear()

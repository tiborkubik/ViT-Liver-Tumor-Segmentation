from abc import ABC, abstractmethod

import torch
from medpy import metric


class VolumeMetric(ABC):

    def __init__(self, name: str, pred_threshold=0.3):
        self.name = name
        self.pred_threshold = pred_threshold
        self.vols_sum = {}
        self.vols_counts = {}

    def update(self, pred_slice: torch.Tensor, target_slice: torch.Tensor, vol_idx: int):
        if vol_idx not in self.vols_sum:
            self.vols_sum[vol_idx] = torch.tensor(0, dtype=torch.float32)
            self.vols_counts[vol_idx] = torch.tensor(0, dtype=torch.long)

        pred_slice = (pred_slice > self.pred_threshold).float()
        pred_slice_sum = torch.sum(pred_slice)
        target_slice_sum = torch.sum(target_slice)

        # Ignore slices where no object is present
        if pred_slice_sum > 0 and target_slice_sum > 0:
            dice_score = self.compute_metric(pred_slice, target_slice)
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

    @abstractmethod
    def compute_metric(self, pred_slice, target_slice):
        pass


class DicePerVolume(VolumeMetric):

    def __init__(self):
        """
        Dice coefficient
        """
        super().__init__('Dice')

    def compute_metric(self, pred_slice, target_slice):
        dice_score = metric.dc(pred_slice.numpy(), target_slice.numpy())
        return dice_score


class VOE(VolumeMetric):

    def __init__(self):
        """
        Volume overlap error = 1 - 	Jaccard coefficient
        """
        super().__init__('VOE')

    def compute_metric(self, pred_slice, target_slice):
        voe_score = 1 - metric.jc(pred_slice.numpy(), target_slice.numpy())
        return voe_score


class MSD(VolumeMetric):

    def __init__(self):
        """
        Mean symmetric surface distance
        """
        super().__init__('MSD')

    def compute_metric(self, pred_slice, target_slice):
        # MSD stands for Maximum symmetric Surface Distance
        msd_score = metric.hd(pred_slice.numpy(), target_slice.numpy())
        return msd_score


class ASSD(VolumeMetric):

    def __init__(self):
        """
        Average symmetric surface distance
        """
        super().__init__('ASSD')

    def compute_metric(self, pred_slice, target_slice):
        assd_score = metric.assd(pred_slice.numpy(), target_slice.numpy())
        return assd_score


class ASD(VolumeMetric):
    """
    Average surface distance metric
    """
    def __init__(self):
        super().__init__('ASD')

    def compute_metric(self, pred_slice, target_slice):
        asd_score = metric.asd(pred_slice.numpy(), target_slice.numpy())
        return asd_score

class RAVD(VolumeMetric):
    """
    Relative absolute volume difference
    """
    def __init__(self):
        super().__init__('RAVD')

    def compute_metric(self, pred_slice, target_slice):
        score = metric.ravd(pred_slice.numpy(), target_slice.numpy())
        return score
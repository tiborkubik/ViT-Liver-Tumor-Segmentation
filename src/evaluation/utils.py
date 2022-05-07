from typing import List
from src.evaluation.metrics import VolumeMetric


def print_metrics(type: str, metrics: List[VolumeMetric]):
    print(type)
    print("==========================")

    for metric in metrics:
        print(F"{metric.name}: {metric.compute_total()}")

    print()


def write_metrics(filename: str, type: str, metrics: List[VolumeMetric]):
    with open(filename, "a") as file:
        file.write(type)
        file.write("\n")

        for metric in metrics:
            file.write(F"{metric.name}: {metric.compute_total()}\n")

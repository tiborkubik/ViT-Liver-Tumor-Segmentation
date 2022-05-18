#!/bin/bash
#PBS -N ViT-Liver-Tumor-Segmentation
#PBS -q gpu
#PBS -l select=1:ngpus=1:gpu_cap=cuda75:cl_adan=True:mem=32gb
#PBS -l walltime=24:00:00

find_in_conda_env() {
conda env list | grep "${@}" >/dev/null 2>/dev/null
}

# Clean up after exit
#trap 'clean_scratch' EXIT

DATADIR=/storage/brno2/home/lakoc/ViT-Liver-Tumor-Segmentation

module add conda-modules-py37

if ! find_in_conda_env "ViT-Liver-Tumor-Segmentation"; then
  conda create -n ViT-Liver-Tumor-Segmentation python=3.9.7
fi

conda activate ViT-Liver-Tumor-Segmentation

echo "ENV created. $(date +"%T") Installing requirements ..."
pip install -r "$DATADIR/requirements.txt"
echo "All packages installed. $(date +"%T")"

ID=4
MODEL="AttentionUNet"
python  "$DATADIR/src/evaluation/evaluate_all_possible_settings.py" -d "$DATADIR/data/dataset/val" -w "$DATADIR/$ID/trained-weights/$MODEL/best-weights.pt" -n $MODEL -sp "$DATADIR/$ID/" -tm 2.5D

echo "Cleaning environment: $(date +"%T")"
conda deactivate

clean_scratch

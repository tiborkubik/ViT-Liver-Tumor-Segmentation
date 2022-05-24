#!/bin/bash
#PBS -N ViT-Liver-Tumor-Segmentation
#PBS -q gpu
#PBS -l select=1:ngpus=1:gpu_cap=cuda75:cl_adan=True:mem=32gb:scratch_local=10gb
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

ID=3
MODEL="TransUNet"

cp -r "$DATADIR/src" "$SCRATCHDIR/src" || {
  echo >&2 "Couldnt copy srcdir to scratchdir."
  exit 2
}

cp -r "$DATADIR/data/dataset/val" "$SCRATCHDIR/dataset" || {
  echo >&2 "Couldnt copy dataset to scratchdir."
  exit 2
}

cp -r "$DATADIR/$ID/trained-weights/$MODEL/best-weights.pt" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy model to scratchdir."
  exit 2
}

BACKBONE="$DATADIR/backbones/imagenet21k_R50+ViT-B_16.npz"
export PYTHONPATH=$DATADIR
python  "$SCRATCHDIR/src/evaluation/evaluate_all_possible_settings.py" -d "$SCRATCHDIR/data/val" -vw $BACKBONE -w "$SCRATCHDIR/best-weights.pt" -n $MODEL -sp $SCRATCHDIR -tm 2D -b 100

cp -r "$SCRATCHDIR/metrics.log" "$DATADIR/$ID" || {
  echo >&2 "Couldnt copy metrics log."
  exit 3
}

echo "Cleaning environment: $(date +"%T")"
conda deactivate

clean_scratch

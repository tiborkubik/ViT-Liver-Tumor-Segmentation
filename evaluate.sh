#!/bin/bash
#PBS -N ViT-Liver-Tumor-Segmentation
#PBS -q gpu
#PBS -l select=1:ngpus=1:gpu_cap=cuda75:cl_adan=True:mem=32gb:scratch_local=80gb
#PBS -l walltime=24:00:00

find_in_conda_env() {
conda env list | grep "${@}" >/dev/null 2>/dev/null
}
qsub -I -q gpu -l select=1:ngpus=1:gpu_cap=cuda75:cl_adan=True:mem=32gb:scratch_local=80gb -l walltime=24:00:00
# Clean up after exit
#trap 'clean_scratch' EXIT
EXPERIMENT_ID=3
MODEL="TransUNet"
TM="2D"
DATADIR=/storage/brno2/home/lakoc/ViT-Liver-Tumor-Segmentation
config=$(<$DATADIR/configs/config"${EXPERIMENT_ID}".txt)
checkpoint="$EXPERIMENT_ID/trained-weights/$MODEL/best-weights.pt"

module add conda-modules-py37

if ! find_in_conda_env "ViT-Liver-Tumor-Segmentation"; then
  conda create -n ViT-Liver-Tumor-Segmentation python=3.9.7
fi

conda activate ViT-Liver-Tumor-Segmentation

echo "ENV created. $(date +"%T") Installing requirements ..."
pip install -r "$DATADIR/requirements.txt"
echo "All packages installed. $(date +"%T")"

# Copy dataset
cp -r "$DATADIR/data" "$SCRATCHDIR/data" || {
  echo >&2 "Couldnt copy dataset to scratchdir."
  exit 2
}

cp -r "$DATADIR/$checkpoint" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy weights."
  exit 2
}

#Copy source codes
cp -r "$DATADIR/src" "$SCRATCHDIR/src" || {
  echo >&2 "Couldnt copy srcdir to scratchdir."
  exit 2
}

# Unzip dataset and split to seg, vol folders
echo "Unzipping dataset: $(date +"%T")..."
unzip "$SCRATCHDIR/data/Training_Batch1.zip" -d "$SCRATCHDIR/data/"
unzip "$SCRATCHDIR/data/Training_Batch2.zip" -d "$SCRATCHDIR/data/"
SUBDIR1="media/nas/01_Datasets/CT/LITS/Training Batch 1"
SUBDIR2="media/nas/01_Datasets/CT/LITS/Training Batch 2"

mkdir "$SCRATCHDIR/data/segs-3d"
mkdir "$SCRATCHDIR/data/vols-3d"

mv "$SCRATCHDIR"/data/"$SUBDIR1"/seg* "$SCRATCHDIR"/data/segs-3d/
mv "$SCRATCHDIR"/data/"$SUBDIR2"/seg* "$SCRATCHDIR"/data/segs-3d/
mv "$SCRATCHDIR"/data/"$SUBDIR1"/vol* "$SCRATCHDIR"/data/vols-3d/
mv "$SCRATCHDIR"/data/"$SUBDIR2"/vol* "$SCRATCHDIR"/data/vols-3d/

echo "Unzipping done: $(date +"%T")"

# Split dataset to train and validation part
echo "Creating validation and train split: $(date +"%T")..."
export PYTHONPATH=$SCRATCHDIR
python "$SCRATCHDIR/src/preprocess/split_dataset.py" -dp "$SCRATCHDIR/data" -v 0.8

echo "Splits successfully created: $(date +"%T")"

# Create 2d slices
echo "Creating 2d slices from dataset: $(date +"%T")..."
python "$SCRATCHDIR/src/preprocess/preprocess_niis.py" -dp "$SCRATCHDIR/data/val"
echo "Slicing done: $(date +"%T")"

BACKBONE="$DATADIR/backbones/imagenet21k_R50+ViT-B_16.npz"
export PYTHONPATH=$DATADIR
python  "$SCRATCHDIR/src/evaluation/evaluate_all_possible_settings.py" -d "$SCRATCHDIR/data/val" -vw $BACKBONE -w "$SCRATCHDIR/best-weights.pt" -n $MODEL -sp $SCRATCHDIR -tm $TM -b 100

cp -r "$SCRATCHDIR/metrics.log" "$DATADIR/$ID" || {
  echo >&2 "Couldnt copy metrics log."
  exit 3
}

echo "Cleaning environment: $(date +"%T")"
conda deactivate

clean_scratch

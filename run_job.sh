#!/bin/bash
#PBS -N ViT-Liver-Tumor-Segmentation
#PBS -q gpu
#PBS -l select=1:ngpus=1:gpu_cap=cuda75:cl_adan=True:mem=32gb:scratch_local=100gb
#PBS -l walltime=24:00:00
#PBS -J 3-5

find_in_conda_env() {
  conda env list | grep "${@}" >/dev/null 2>/dev/null
}

# Clean up after exit
#trap 'clean_scratch' EXIT
config=$(<$DATADIR/configs/config"${PBS_ARRAY_INDEX}".txt)

DATADIR=/storage/brno2/home/lakoc/ViT-Liver-Tumor-Segmentation

echo "$PBS_JOBID is running on node $(hostname -f) in a scratch directory $SCRATCHDIR with following config: $config. Time: $(date +"%T")"

#Copy source codes
echo "Copying ENV. $(date +"%T")"
cp -r "$DATADIR/requirements.txt" "$SCRATCHDIR" || {
  echo >&2 "Couldnt copy srcdir to scratchdir."
  exit 2
}

module add conda-modules-py37

if ! find_in_conda_env "ViT-Liver-Tumor-Segmentation"; then
  conda create -n ViT-Liver-Tumor-Segmentation python=3.9.7
fi

conda activate ViT-Liver-Tumor-Segmentation

echo "ENV created. $(date +"%T") Installing requirements ..."
pip install -r "$SCRATCHDIR/requirements.txt"
echo "All packages installed. $(date +"%T")"

echo "Copying data from FE: $(date +"%T").."

# Copy dataset
cp -r "$DATADIR/data" "$SCRATCHDIR/data" || {
  echo >&2 "Couldnt copy dataset to scratchdir."
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
python "$SCRATCHDIR/src/preprocess/split_dataset.py" -d "$SCRATCHDIR/data"

echo "Splits successfully created: $(date +"%T")"

# Create 2d slices
echo "Creating 2d slices from dataset: $(date +"%T")..."
python "$SCRATCHDIR/src/preprocess/preprocess_niis.py" -dp "$SCRATCHDIR/data/train"
python "$SCRATCHDIR/src/preprocess/preprocess_niis.py" -dp "$SCRATCHDIR/data/val"
echo "Slicing done: $(date +"%T")"

# Start training
mkdir "$SCRATCHDIR/trained-weights"
mkdir "$SCRATCHDIR/documentation"
echo "All ready. Starting trainer: $(date +"%T")"
BACKBONE="$DATADIR/backbones/imagenet21k_R50+ViT-B_16.npz"

python3 "$SCRATCHDIR/src/trainer/train.py" -dt "$SCRATCHDIR/data/train" -dv "$SCRATCHDIR/data/val" -sp $SCRATCHDIR -vw $BACKBONE $config

echo "Cleaning environment: $(date +"%T")"
conda deactivate

echo "Training done. Copying back to FE: $(date +"%T")"
mkdir "$DATADIR/$PBS_ARRAY_INDEX"
# Copy data back to FE
cp -r "$SCRATCHDIR/documentation" "$DATADIR/$PBS_ARRAY_INDEX" || {
  echo >&2 "Couldnt copy documentation to datadir."
  exit 3
}
cp -r "$SCRATCHDIR/trained-weights" "$DATADIR/$PBS_ARRAY_INDEX" || {
  echo >&2 "Couldnt copy weights to datadir."
  exit 3
}

cp -r "$SCRATCHDIR/metrics.log" "$DATADIR/$PBS_ARRAY_INDEX" || {
  echo >&2 "Couldnt copy metrics log."
  exit 3
}

clean_scratch

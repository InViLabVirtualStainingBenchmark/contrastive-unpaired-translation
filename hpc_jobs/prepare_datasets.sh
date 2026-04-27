#!/bin/bash
# prepare_datasets.sh
# Run once manually on the login node after datasets have been transferred via rsync.
# Creates symlinked trainA/trainB/testA/testB folders on scratch for CUT.
# CUT expects separate A and B folders -- no combine_A_and_B.py step is needed.
#
# Expected dataset layout after rsync:
#   $VSC_SCRATCH/dataset/BCI/
#     HE/train/   HE/test/
#     IHC/train/  IHC/test/
#
#   $VSC_SCRATCH/dataset/MIST-HER2/
#     trainA/  trainB/  valA/  valB/
#
# Usage:
#   bash prepare_datasets.sh
#
# Do NOT submit this with sbatch.

set -euo pipefail

echo "=== Verifying source dataset folders ==="

for DIR in \
    "$VSC_SCRATCH/dataset/BCI/HE/train" \
    "$VSC_SCRATCH/dataset/BCI/HE/test" \
    "$VSC_SCRATCH/dataset/BCI/IHC/train" \
    "$VSC_SCRATCH/dataset/BCI/IHC/test" \
    "$VSC_SCRATCH/dataset/MIST-HER2/trainA" \
    "$VSC_SCRATCH/dataset/MIST-HER2/trainB" \
    "$VSC_SCRATCH/dataset/MIST-HER2/valA" \
    "$VSC_SCRATCH/dataset/MIST-HER2/valB"
do
    if [ ! -d "$DIR" ]; then
        echo "ERROR: Missing folder: $DIR"
        echo "Run the rsync transfer from your local machine first."
        exit 1
    fi
    echo "  OK: $DIR"
done

echo ""
echo "=== Cleaning up Zone.Identifier files from Windows transfer ==="
find "$VSC_SCRATCH/dataset/BCI" -name "*.Identifier" -delete && echo "  BCI cleaned"
find "$VSC_SCRATCH/dataset/MIST-HER2" -name "*.Identifier" -delete && echo "  MIST-HER2 cleaned"

echo ""
echo "=== Creating CUT symlink folders on scratch ==="

# BCI
mkdir -p "$VSC_SCRATCH/cut-BCI"
ln -sfn "$VSC_SCRATCH/dataset/BCI/HE/train"  "$VSC_SCRATCH/cut-BCI/trainA"
ln -sfn "$VSC_SCRATCH/dataset/BCI/IHC/train" "$VSC_SCRATCH/cut-BCI/trainB"
ln -sfn "$VSC_SCRATCH/dataset/BCI/HE/test"   "$VSC_SCRATCH/cut-BCI/testA"
ln -sfn "$VSC_SCRATCH/dataset/BCI/IHC/test"  "$VSC_SCRATCH/cut-BCI/testB"
echo "  cut-BCI symlinks created"

# MIST-HER2 (val split is used as test)
mkdir -p "$VSC_SCRATCH/cut-MIST-HER2"
ln -sfn "$VSC_SCRATCH/dataset/MIST-HER2/trainA" "$VSC_SCRATCH/cut-MIST-HER2/trainA"
ln -sfn "$VSC_SCRATCH/dataset/MIST-HER2/trainB" "$VSC_SCRATCH/cut-MIST-HER2/trainB"
ln -sfn "$VSC_SCRATCH/dataset/MIST-HER2/valA"   "$VSC_SCRATCH/cut-MIST-HER2/testA"
ln -sfn "$VSC_SCRATCH/dataset/MIST-HER2/valB"   "$VSC_SCRATCH/cut-MIST-HER2/testB"
echo "  cut-MIST-HER2 symlinks created"

echo ""
echo "=== Image counts ==="
echo "  cut-BCI/trainA : $(find "$VSC_SCRATCH"/cut-BCI/trainA -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-BCI/trainB : $(find "$VSC_SCRATCH"/cut-BCI/trainB -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-BCI/testA  : $(find "$VSC_SCRATCH"/cut-BCI/testA  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-BCI/testB  : $(find "$VSC_SCRATCH"/cut-BCI/testB  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-MIST-HER2/trainA : $(find "$VSC_SCRATCH"/cut-MIST-HER2/trainA -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-MIST-HER2/trainB : $(find "$VSC_SCRATCH"/cut-MIST-HER2/trainB -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-MIST-HER2/testA  : $(find "$VSC_SCRATCH"/cut-MIST-HER2/testA  -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "  cut-MIST-HER2/testB  : $(find "$VSC_SCRATCH"/cut-MIST-HER2/testB  -maxdepth 1 \( -type f -o -type l \) | wc -l)"

echo ""
echo "Done. Next step: sbatch install.sh"

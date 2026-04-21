# DOCUMENT.md

<!--
This file lives in the root of every forked repo.
Fill it in as you go. Do not reconstruct it after the fact.
Keep entries factual and brief. The audience is a future person
reproducing your setup on a different machine or the HPC cluster.
-->

---

## Model Info

<!--
Copy this information from the upstream repo's README and paper.
"Paired or unpaired" refers to whether the model assumes paired training data.
If the model is domain-specific to virtual staining, note the exact staining task (e.g. H&E to HER2 IHC).
-->

- **Model name:** Contrastive Unpaired Translation (CUT)
- **Upstream repo URL:** https://github.com/taesungp/contrastive-unpaired-translation
- **Fork URL:** https://github.com/InViLabVirtualStainingBenchmark/contrastive-unpaired-translation
- **Upstream last commit date:** Sep 5, 2023
- **Paper / citation:** Contrastive Learning for Unpaired Image-to-Image Translation https://arxiv.org/pdf/2007.15651
- **Paired or unpaired assumption:** Unpaired
- **Intended staining task (if domain-specific):** general image-to-timage translation

---

## Environment Claimed by Authors

<!--
Record exactly what the authors say in their README or requirements file.
Do not adjust or interpret -- copy their stated versions.
"Requirements file present" should note the filename if it exists.
If no version is specified for Python or PyTorch, write "not specified".
-->

- **Python version:** 3.6
- **PyTorch version:** 1.4.0
- **CUDA version:** not specified
- **Installation method:** pip + conda
- **Requirements file present:** requirements.txt and environment.yml 
- **Pretrained weights available:** yes
- **Pretrained weights notes:** hosted at http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar -- personal/lab server, at-risk link likely to rot
<!-- Where are they hosted? Are they behind a login? Is the link likely to rot (GDrive, Dropbox, personal server)? -->

---

## Environment Actually Used

<!--
Record the environment you actually created and tested in.
If you deviated from what the authors specified, briefly note why (e.g. "authors' version not compatible with CUDA 12.1").
Conda env name should follow the convention: the model's short name.
-->

- **Python version:** 3.8.20
- **PyTorch version:** 1.13.1
- **CUDA version:** 12.1 (driver) / 11.7 (PyTorch build)
- **Conda environment name:** cut
- **Date tested:** 21-04-2026
- **Hardware:** RTX 4090, WSL2 on Windows 11

---

## Installation

<!--
Follow the authors' README exactly before making any changes.
Record the commands you ran in order.
If an error occurred, paste the key line of the error (not the full traceback) and then record the fix.
If installation succeeded without issues, write "No issues."
-->

### Commands Run

```bash
# paste the installation commands here in order
conda create -n cut python=3.8 -y
conda activate cut
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Issues and Fixes

<!--
Format: problem encountered -> fix applied.
If no issues, write "None."
-->

| Issue | Fix Applied |
| --- | --- |
| Authors specify Python 3.6 / PyTorch 1.4.0, incompatible with CUDA 12.1 | Used Python 3.8.20 / PyTorch 1.13.1 with pytorch-cuda=11.7 (11.7 build runs under 12.1 driver) |
| Evaluation dependencies (torchmetrics, lpips, torch-fidelity) not installed in model env | Kept separate, evaluation runs via the shared vs-benchmark conda env and evaluate.py |
| libcuda.so not found on first inference run | export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH before running |
| ~/miniconda3/envs/cut/etc/conda/activate.d/cuda_wsl.sh (new file) | Added conda activate hook to export LD_LIBRARY_PATH=/usr/lib/wsl/lib | libcuda.so not found at runtime under WSL2 without this fix |

### GPU Confirmation

<!--
Paste the output of the check below so there is proof the GPU was visible.
Command: python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
-->

```
1.13.1 True NVIDIA GeForce RTX 4090
```

---

## Dataset Preparation

<!--
Record how the dataset was prepared for this specific model.
"Format expected" means what folder layout or file structure the model's data loader assumes
(e.g. side-by-side paired images, separate A/B folders, CSV manifest, etc.).
"Conversion applied" means any script or command you ran to reformat the standard BCI/MIST-HER2
download into the format this model needs.
If no conversion was needed, write "None -- dataset used as downloaded."
-->

- **Dataset used:** both
- **Format expected by model:** separate trainA/ trainB/ testA /testB folders
- **Conversion applied:** 
    - HE -> trainA/testA, IHC -> trainB/testB via rsync excluding Zone.Identifier files.
    - BCI testB had 262 extra IHC images with no HE counterpart -- removed to ensure stem-matched evaluation.
    - Mismatch was present in the original dataset, not introduced by conversion.
    
    ```bash
    # paste conversion command(s) here if any
    # BCI -- create folder structure

    mkdir -p ~/internship-models/datasets/BCI-cut/trainA
    mkdir -p ~/internship-models/datasets/BCI-cut/trainB
    mkdir -p ~/internship-models/datasets/BCI-cut/testA
    mkdir -p ~/internship-models/datasets/BCI-cut/testB

    # BCI -- copy images excluding Zone.Identifier metadata files

    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/BCI_dataset/BCI_dataset/HE/train/ ~/internship-models/datasets/BCI-cut/trainA/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/BCI_dataset/BCI_dataset/IHC/train/ ~/internship-models/datasets/BCI-cut/trainB/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/BCI_dataset/BCI_dataset/HE/test/ ~/internship-models/datasets/BCI-cut/testA/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/BCI_dataset/BCI_dataset/IHC/test/ ~/internship-models/datasets/BCI-cut/testB/

    # BCI -- remove 262 extra IHC test images that have no HE counterpart

    comm -23 <(ls ~/internship-models/datasets/BCI-cut/testB | sort) <(ls ~/internship-models/datasets/BCI-cut/testA | sort) | xargs -I{} rm ~/internship-models/datasets/BCI-cut/testB/{}

    # MIST-HER2 -- create folder structure

    mkdir -p ~/internship-models/datasets/MIST-HER2-cut/trainA
    mkdir -p ~/internship-models/datasets/MIST-HER2-cut/trainB
    mkdir -p ~/internship-models/datasets/MIST-HER2-cut/testA
    mkdir -p ~/internship-models/datasets/MIST-HER2-cut/testB

    # MIST-HER2 -- copy images, map valA/valB -> testA/testB, exclude Zone.Identifier

    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/MIST/HER2-004/TrainValAB/trainA/ ~/internship-models/datasets/MIST-HER2-cut/trainA/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/MIST/HER2-004/TrainValAB/trainB/ ~/internship-models/datasets/MIST-HER2-cut/trainB/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/MIST/HER2-004/TrainValAB/valA/ ~/internship-models/datasets/MIST-HER2-cut/testA/
    rsync -av --exclude='*Zone.Identifier' ~/internship-models/datasets/original/MIST/HER2-004/TrainValAB/valB/ ~/internship-models/datasets/MIST-HER2-cut/testB/
    ```
    
- **Final folder layout used:**
    
    ```
    dataset/BCI-cut/
        trainA/   -- 3896 H&E train images
        trainB/   -- 3896 IHC train images
        testA/    -- 715 H&E test images
        testB/    -- 715 IHC test images (262 extra IHC-only images removed)
    dataset/MIST-HER2-cut/
        trainA/   -- 4642 H&E train images (val->test, Zone.Identifier excluded)
        trainB/   -- 4642 IHC train images
        testA/    -- 1000 H&E test images
        testB/    -- 1000 IHC test images
    ```
    
- **Number of images used for smoke test (train / test):** 200 train / 20 test (subset, see Phase 3B)
---

## Pretrained Weights

<!--
Only fill this section if pretrained weights exist.
Record the exact download source. Flag any link that is not on a stable host
(Zenodo and HuggingFace are stable; Google Drive, Dropbox, and personal servers are at risk).
Record where you placed the weights relative to the repo root.
-->

- **Download source URL:**  http://efrosgans.eecs.berkeley.edu/CUT/pretrained_models.tar
- **Host stability:**  at-risk (personal/lab server at Berkeley)
- **Weights placed at (relative path):** checkpoints/horse2zebra_cut_pretrained/latest_net_G.pth
- **Size on disk:** 261MB (full tar, contains 6 models: horse2zebra, cat2dog, cityscapes -- CUT and FastCUT variants each)
- **Note:** No histopathology pretrained weights available. Inference smoke test uses horse2zebra_cut_pretrained to verify the code path only. Benchmark results come from training on BCI and MIST-HER2.

---

## Inference Smoke Test

<!--
Run inference before training if pretrained weights are available -- it is faster
and confirms the code path works independently of the training loop.
Use 10-20 images from the BCI or MIST-HER2 test split.
"Visual check" is a qualitative sanity check only -- not a metric.
Valid outcomes: "images look like expected domain", "blank/grey output", "wrong resolution", "file not written".
-->

- **Script / command run:**
    
    ```bash
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    
    python test.py --dataroot ~/internship-models/datasets/BCI-cut --name horse2zebra_cut_pretrained --CUT_mode CUT --phase test --num_test 20 --results_dir ~/internship-models/results/cut-inference-test
    ```
    
- **Output folder:** ~/internship-models/results/cut-inference-test/horse2zebra_cut_pretrained/test_latest/images/fake_B
- **Number of output images produced:** 20
- **Output image dimensions:** 256x256
- **Visual check result:** okay
- **Time to run (approx):** < 1 min
- **Errors or warnings during inference:**
    - libcudnn_cnn_infer.so.8 not found on first run (fixed by setting LD_LIBRARY_PATH=/usr/lib/wsl/lib)
    - torchvision interpolation deprecation warnin (harmless)
<!-- "None" if clean. Otherwise paste the key error line. -->

---

## Training Smoke Test

<!--
Run training for 5 epochs minimum. The goal is a clean exit, not a useful model.
Use the smallest viable batch size and the model's default resolution unless that causes an OOM error.
Always set checkpoint saving to every epoch (e.g. --save_epoch_freq 1 for pix2pix-style repos)
so there is proof a checkpoint was written.
Monitor GPU memory with: watch -n 1 nvidia-smi (run in a second terminal).
-->

- **Script / command run:**
    
    ```bash
    python train.py --dataroot ~/internship-models/datasets/BCI-cut-smoke --name cut_bci_smoke --CUT_mode CUT --n_epochs 5 --n_epochs_decay 0 --save_epoch_freq 1 --checkpoints_dir ~/internship-models/results/cut-smoke-train/checkpoints --gpu_ids 0
    ```
    
- **Dataset used:** BCI-cut-smoke (200 train pairs, 20 test pairs)
- **Epochs run:** 5
- **Batch size:** 1
- **Input resolution:** 256x256 (default)
- **Time per epoch (approx):** 20-30 seconds
- **Peak GPU memory (approx, from nvidia-smi):** ~20,436 MiB (~20GB)
- **Checkpoint saved:** yes 
- **Checkpoint path:** ~/internship-models/results/cut-smoke-train/checkpoints/cut_bci_smoke/
- **Resolution note:** CUT default output is 256x256. GT images are 1024x1024.
Full benchmark run must add --load_size 1024 --crop_size 1024 to match native resolution.
- **Crash or error during training:** None (torchvision interpolation deprecation warning only (harmless))
<!-- "None" if clean. Otherwise paste the key error line and the fix applied. -->

## Smoke Test Metrics (BCI, 20 pairs, 5 epochs -- not benchmark numbers)
| Metric | Mean | Std |
| --- | --- | --- |
| PSNR | 14.15 dB | 5.59 |
| SSIM | 0.656 | 0.098 |
| MS-SSIM | 0.510 | 0.107 |
| LPIPS (AlexNet) | 0.728 | 0.106 |
| LPIPS (VGG) | 0.699 | 0.058 |
| MAE | 0.218 | 0.117 |
| FID | 272.44 | -- |
---

## Output Verification

<!--
Open 3-5 output images and compare them visually against the ground-truth target.
This is not a metric -- just a check that the model produced something in the right domain.
"Expected domain" for BCI would be IHC HER2-stained tissue with brown DAB staining on a light background.
Record one or two example output filenames so the check is reproducible.
-->

- **Output folder:** ~/internship-models/results/cut-smoke-train/test_output/cut_bci_smoke/test_latest/images/fake_B
- **Example output filenames:** 00268_test_2+.png, 00269_test_3+.png
- **Dimensions match input:** yes (256x256)
- **Visual sanity check:** 
    - outputs show IHC-like light background with tissue structures spatially preserved and roughly aligned with H&E input.
    - Brown DAB signal is present but very faint (consistent with only 5 epochs on 200 images).
    - Domain shift is occurring in the correct direction.
<!-- e.g. "outputs show IHC-like staining, structures roughly aligned with H&E input" -->
- **Any obvious artifacts or failure modes:** DAB staining intensity significantly weaker than ground truth (expected at this training scale, not a code issue).

---

## Changes Made to Original Code

<!--
Record every change made to the original repo, no matter how small.
Do not make changes that alter model architecture or training logic.
Only changes needed for the code to run in the benchmark environment are allowed.
Add rows as needed.
-->

| File | Change Description | Reason |
| --- | --- | --- |
|  |  |  |
|  |  |  |

<!--
Common examples of acceptable changes:

- Pinning a dependency version in requirements.txt (e.g. torch==2.1.0) because no version was specified
- Replacing a hardcoded absolute path with a command-line argument
- Removing an import that is not used and is not installable in the benchmark environment
- Adapting the data loader to accept BCI/MIST-HER2 folder structure
-->

---

## Frozen Environment

<!--
After the smoke test passes, export and commit the environment file.
Command: conda env export > environment_<model-name>.yml
This file is what gets adapted for the HPC migration later.
Note any packages that are unusual, very large, or likely to cause conflicts on the cluster.
-->

- **Environment file:** `environment_<model-name>.yml`
- **Committed to fork:** yes / no
- **Notes on unusual or heavy dependencies:**
<!-- e.g. "requires openslide-python which needs a system-level apt install" -->

---

## HPC Readiness Notes

<!--
Fill this in after the local smoke test passes.
Flag anything that will need attention before running on the VSC cluster.
Common issues: GUI/display dependencies (matplotlib backends), hardcoded CUDA package versions,
dependencies that require apt/system installs, very large model downloads.
Leave blank until local test is complete.
-->

- **Display/GUI dependencies to remove or neutralize:**
- **System-level dependencies (non-pip/conda):**
- **Estimated GPU memory requirement:**
- **Estimated storage requirement (weights + data):**
- **Other notes for cluster adaptation:**

---

## Summary

<!--
Write 2-4 sentences summarizing what worked, what did not, and what the next step is.
Be specific. Include the overall pass/fail verdict.
This is the first thing someone reads when picking this model back up.
-->

**Overall result:** PASS / FAIL / PARTIAL

<!-- Example pass:
"[Model] smoke test completed on [date]. Inference with pretrained weights passed on 10 BCI test images.
Training ran for 5 epochs without crash. One change was made to the data loader to accept separate
source/target folders. Frozen environment committed. Ready for full benchmark run."

Example fail:
"[Model] smoke test failed at the environment step. The required PyTorch version (1.4) is not
compatible with CUDA 12.1. Blocked until a workaround is identified. Do not schedule for HPC."
-->
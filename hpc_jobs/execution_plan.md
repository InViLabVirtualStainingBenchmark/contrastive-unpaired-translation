# CUT -- Cluster Execution Plan

Complete reference for running CUT on VSC Tier 2 Antwerp.
All scripts live in `hpc_jobs/` inside this repo.
Run all commands from the cluster login node unless stated otherwise.

---

## Cluster quick reference

| Property                 | Value                                                                        |
|--------------------------|------------------------------------------------------------------------------|
| Login node               | `login.hpc.uantwerpen.be`                                                    |
| Vaughan login node       | `login-vaughan.hpc.uantwerpen.be`                                            |
| SSH (WSL / Linux)        | `ssh vsc<YOUR-ID>@login.hpc.uantwerpen.be`                                   |
| SSH (Windows PowerShell) | `ssh -i C:\Users\marku\.ssh\id_ed25519 vsc<YOUR-ID>@login.hpc.uantwerpen.be` |
| Account                  | `ap_invilab_td_thesis`                                                       |
| Partition                | `ampere_gpu`                                                                 |
| Max wall time            | 24 hours (`1-00:00:00`)                                                      |
| GPU nodes in partition   | 1 node (`nvam1.vaughan`)                                                     |
| Persistent storage       | `$VSC_DATA`                                                                  |
| Fast scratch (datasets)  | `$VSC_SCRATCH`                                                               |
| PyTorch module           | `PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1`                                |
| Python version (module)  | 3.9.25                                                                       |

---

## Script inventory

| Script                         | Type              | What it does                                                                                                                                                                                                   |
|--------------------------------|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `setup_project.sh`             | bash (manual)     | Creates folder tree under `$VSC_DATA` and `$VSC_SCRATCH`                                                                                                                                                       |
| `prepare_datasets.sh`          | bash (manual)     | Creates `cut-BCI` and `cut-MIST-HER2` symlink folders on scratch                                                                                                                                               |
| `install.sh`                   | sbatch            | Builds CUT model venv, pip-installs CUT deps, runs sanity checks                                                                                                                                               |
| `install_eval.sh`              | bash (login node) | **Lives in the evaluate repo** (`$VSC_DATA/evaluate/hpc_jobs/`). Builds shared eval venv, pre-downloads LPIPS and Cellpose weights. Shared across all models. Run directly on the login node — not via sbatch. |
| `train_validate.sh`            | sbatch            | 5-epoch BCI run -- confirmation gate before full runs                                                                                                                                                          |
| `train_BCI_full_e400.sh`       | sbatch            | Full 400-epoch CUT training on BCI                                                                                                                                                                             |
| `train_MIST-HER2_full_e400.sh` | sbatch            | Full 400-epoch CUT training on MIST-HER2                                                                                                                                                                       |
| `infer_BCI_full.sh`            | sbatch            | Inference on full BCI test split                                                                                                                                                                               |
| `infer_MIST-HER2_full.sh`      | sbatch            | Inference on full MIST-HER2 test split                                                                                                                                                                         |
| `eval_BCI_full.sh`             | sbatch            | Runs evaluate.py on BCI predictions using shared eval venv                                                                                                                                                     |
| `eval_MIST-HER2_full.sh`       | sbatch            | Runs evaluate.py on MIST-HER2 predictions using shared eval venv                                                                                                                                               |

---

## Execution order

### Phase A -- One-time setup (run manually, no sbatch)

**Step A1 -- Connect and verify**

From WSL terminal:
```bash
ssh vsc<YOUR-ID>@login.hpc.uantwerpen.be
echo $VSC_DATA
echo $VSC_SCRATCH
```

Expected:
- `$VSC_DATA`   = `/data/antwerpen/212/vsc21212`
- `$VSC_SCRATCH` = `/scratch/antwerpen/212/vsc21212`

**Step A2 -- Create the bare minimum to allow cloning**

```bash
mkdir -p $VSC_DATA/projects/cut/code
```

**Step A3 -- Clone the CUT repo on the cluster**

```bash
cd $VSC_DATA/projects/cut/code
git clone https://github.com/InViLabVirtualStainingBenchmark/contrastive-unpaired-translation cut
```

The `hpc_jobs/` folder is now available at `$VSC_DATA/projects/cut/code/cut/hpc_jobs/`.

**Step A4 -- Create the full project folder tree**

```bash
bash $VSC_DATA/projects/cut/code/cut/hpc_jobs/setup_project.sh
```

Expected output ends with: `Done. Next step: transfer datasets via rsync, then run prepare_datasets.sh`

**Step A5 -- Clone the evaluate repository**

```bash
git clone https://github.com/InViLabVirtualStainingBenchmark/evaluate.git $VSC_DATA/evaluate
```

Also create the eval logs folder:
```bash
mkdir -p $VSC_DATA/evaluate/logs
```

Verify:
```bash
ls $VSC_DATA/evaluate
```
Expected: `evaluate.py  environment.yml  hpc_jobs  README.md`

```bash
ls $VSC_DATA/evaluate/hpc_jobs
```
Expected: `cluster_plan.md  install_eval.sh`

**Step A6 -- Transfer datasets via WinSCP (run from the lab PC, not the cluster)**

Connect WinSCP with:
- Host: `login.hpc.uantwerpen.be`
- Port: `22`
- Username: `vsc<YOUR-ID>`
- Private key: `C:\Users\<USER>\.ssh\id_ed25519`

Transfer 1 -- BCI:
- Local source: navigate into `C:\...\BCI_dataset\` so you see `HE\` and `IHC\`
- Remote destination: `/scratch/antwerpen/212/vsc<YOUR-ID>/dataset/BCI/`
- Drag both `HE` and `IHC` folders across

Transfer 2 -- MIST-HER2:
- Local source: navigate into `C:\...\HER2\TrainValAB\` so you see `trainA\`, `trainB\`, `valA\`, `valB\`
- Remote destination: `/scratch/antwerpen/212/vsc<YOUR-ID>/dataset/MIST-HER2/`
- Drag all four folders across

Verify on the cluster after transfer:
```bash
ls $VSC_SCRATCH/dataset/BCI/
ls $VSC_SCRATCH/dataset/MIST-HER2/
```
Expected: `HE  IHC` and `trainA  trainB  valA  valB`

**Step A7 -- Create dataset symlinks**

```bash
bash $VSC_DATA/projects/cut/code/cut/hpc_jobs/prepare_datasets.sh
```

Verify symlinks work (use `ls | wc -l`, not `find`, because find does not follow symlinks without `-L`):
```bash
ls $VSC_SCRATCH/cut-BCI/trainA | wc -l
ls $VSC_SCRATCH/cut-BCI/trainB | wc -l
ls $VSC_SCRATCH/cut-MIST-HER2/trainA | wc -l
ls $VSC_SCRATCH/cut-MIST-HER2/trainB | wc -l
```
Expected: non-zero counts for all four (BCI: 3896, MIST-HER2: 4642).

---

### Phase B -- Environment install (sbatch)

Both install jobs are independent and can be submitted at the same time.

**Step B1 -- Install environments**

CUT model venv (sbatch job — no internet required, modules handle PyTorch):
```bash
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/install.sh
```

Shared eval venv (run directly on login node — requires internet for weight downloads):
```bash
bash $VSC_DATA/evaluate/hpc_jobs/install_eval.sh
```

**Step B2 -- Monitor**

```bash
squeue -u $USER
```

Note: the `ampere_gpu` partition has only one GPU node. Jobs may sit in `PD` (pending)
for hours depending on cluster load. Check the node state with:
```bash
sinfo -p ampere_gpu
```
A state of `mix-` means the node is draining and will not accept new jobs until current
ones finish. This is normal -- just wait.

**Step B3 -- Verify both logs after jobs complete**

```bash
cat $VSC_DATA/projects/cut/logs/install.<jobid>.out
cat $VSC_DATA/evaluate/logs/install_eval.<jobid>.out
```

Gate: both logs must end with `All checks passed` before submitting anything else.
If any import check fails, fix the relevant pip install block and resubmit that job only.

---

### Phase C -- Cluster confirmation gate (sbatch)

Only submit after both Phase B jobs have passed.

**Step C1 -- Submit validation training job**

```bash
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/train_validate.sh
```

**Step C2 -- Monitor**

```bash
squeue -u $USER
tail -f $VSC_DATA/projects/cut/logs/train_validate.<jobid>.out
```

**Step C3 -- Verify pass criteria**

All four must be true before submitting full runs:

1. Job log exits without a Python traceback.
2. Loss values printed in the log are not NaN.
3. Checkpoint files exist:
   ```bash
   find $VSC_DATA/projects/cut/outputs/checkpoints/BCI_smoke_e5 -name "*.pth" | sort
   ```
   Expected: 5 `.pth` files (one per epoch, since `--save_epoch_freq 1`).
4. GPU log has entries:
   ```bash
   tail -5 $VSC_DATA/projects/cut/logs/gpu_train_validate.csv
   ```
   Expected: non-zero GPU utilization values.

**Step C4 -- Calibrate time-per-epoch for full training jobs**

Check the validate job wall time from the log timestamps, divide by 5 epochs,
multiply by 400, add 20% margin. The ampere_gpu partition has a hard maximum of
24 hours (`1-00:00:00`). If the full run exceeds 24 hours, job chaining will be
needed -- discuss with Thomas before submitting full runs.

Update `--time` in `train_BCI_full_e400.sh` and `train_MIST-HER2_full_e400.sh`
before submitting Phase D.

---

### Phase D -- Full training (sbatch)

Only one GPU node is available so both jobs will queue sequentially, not in parallel.
Submit both anyway -- the scheduler will run them one after the other automatically.

```bash
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/train_BCI_full_e400.sh
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/train_MIST-HER2_full_e400.sh
```

Monitor progress:
```bash
squeue -u $USER

# BCI log
tail -f $VSC_DATA/projects/cut/logs/train_BCI.<jobid>.out

# MIST log
tail -f $VSC_DATA/projects/cut/logs/train_MIST.<jobid>.out

# Checkpoint progress
find $VSC_DATA/projects/cut/outputs/checkpoints -name "*.pth" | sort
```

Checkpoints are saved every 50 epochs. After 200 epochs you will also see `latest_net_G.pth`.

If a job is cancelled due to TIMEOUT: increase `--time` by 25% and resubmit with
`--continue` or by resuming from the latest checkpoint. Never reduce epoch count.

---

### Phase E -- Inference (sbatch)

Submit inference for each dataset as soon as that dataset's training job completes.

```bash
# After train_BCI_full_e400.sh finishes:
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/infer_BCI_full.sh

# After train_MIST-HER2_full_e400.sh finishes:
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/infer_MIST-HER2_full.sh
```

Verify output image count after each job:
```bash
find $VSC_DATA/projects/cut/outputs/results/BCI_full_e400/test_latest/images/fake_B -name "*.png" | wc -l
find $VSC_DATA/projects/cut/outputs/results/MIST-HER2_full_e400/test_latest/images/fake_B -name "*.png" | wc -l
```
Expected: BCI ~3896 images, MIST-HER2 ~4642 images (one PNG per test image).

---

### Phase F -- Evaluation (sbatch)

Uses the shared eval venv at `$VSC_DATA/evaluate/venv_eval/`.
Submit after the corresponding inference job has completed.

```bash
# After infer_BCI_full.sh finishes:
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/eval_BCI.sh

# After infer_MIST-HER2_full.sh finishes:
sbatch $VSC_DATA/projects/cut/code/cut/hpc_jobs/eval_MIST-HER2.sh
```

Both jobs append rows to `$VSC_DATA/benchmark_results.csv`.

Verify results:
```bash
cat $VSC_DATA/benchmark_results.csv
```

---

### Phase G -- Transfer results back to local (run from the lab PC)

From Windows PowerShell:
```bash
--
```

Open `benchmark_results_cluster.csv`, copy the two CUT rows (BCI and MIST-HER2)
into the shared benchmark spreadsheet (`vs_benchmark_triage.xlsx`).
Update the kanban board column for CUT from Smoke tested to Eval done.

---

## Key paths on the cluster

| Artifact             | Path                                                                                    |
|----------------------|-----------------------------------------------------------------------------------------|
| Repo                 | `$VSC_DATA/projects/cut/code/cut/`                                                      |
| Model venv           | `$VSC_DATA/projects/cut/venv_cut/`                                                      |
| Shared eval venv     | `$VSC_DATA/evaluate/venv_eval/`                                                         |
| Job scripts          | `$VSC_DATA/projects/cut/code/cut/hpc_jobs/`                                             |
| Slurm logs           | `$VSC_DATA/projects/cut/logs/`                                                          |
| Eval install log     | `$VSC_DATA/evaluate/logs/`                                                              |
| BCI dataset (raw)    | `$VSC_SCRATCH/dataset/BCI/`                                                             |
| MIST-HER2 (raw)      | `$VSC_SCRATCH/dataset/MIST-HER2/`                                                       |
| BCI symlinks         | `$VSC_SCRATCH/cut-BCI/`                                                                 |
| MIST-HER2 symlinks   | `$VSC_SCRATCH/cut-MIST-HER2/`                                                           |
| BCI checkpoints      | `$VSC_DATA/projects/cut/outputs/checkpoints/BCI_full_e400/`                             |
| MIST checkpoints     | `$VSC_DATA/projects/cut/outputs/checkpoints/MIST-HER2_full_e400/`                       |
| BCI results          | `$VSC_DATA/projects/cut/outputs/results/BCI_full_e400/test_latest/images/fake_B/`       |
| MIST results         | `$VSC_DATA/projects/cut/outputs/results/MIST-HER2_full_e400/test_latest/images/fake_B/` |
| Evaluate repo        | `$VSC_DATA/evaluate/`                                                                   |
| evaluate.py          | `$VSC_DATA/evaluate/evaluate.py`                                                        |
| Shared metrics table | `$VSC_DATA/benchmark_results.csv`                                                       |

---

## Monitoring commands

```bash
# Check all running and queued jobs
squeue -u $USER

# Check GPU node state
sinfo -p ampere_gpu

# Get detailed job info including estimated start time
scontrol show job <jobid>

# Watch a log file live
tail -f $VSC_DATA/projects/cut/logs/train_BCI.<jobid>.out

# Check GPU utilization during training
tail -5 $VSC_DATA/projects/cut/logs/gpu_train_BCI.csv

# Find all checkpoints
find $VSC_DATA/projects/cut/outputs/checkpoints -name "*.pth" | sort

# Count inference output images
find $VSC_DATA/projects/cut/outputs/results -name "*.png" | wc -l
```

---

## Common issues

| Problem                                   | Cause                                          | Fix                                                                               |
|-------------------------------------------|------------------------------------------------|-----------------------------------------------------------------------------------|
| Job stuck in PD (Resources)               | Node draining (`mix-` state) or queue priority | Check `sinfo -p ampere_gpu` and `scontrol show job <id>` for estimated start time |
| Job cancelled: TIMEOUT                    | Time limit too low                             | Increase `--time` by 25%, resubmit. Never reduce epoch count.                     |
| ModuleNotFoundError                       | Package missing from venv                      | Add to pip install block in `install.sh`, resubmit install job                    |
| Checkpoint not written                    | Training crashed                               | Check log for Python traceback, fix and resubmit                                  |
| PRED_DIR not found in eval job            | Inference not complete or path wrong           | Confirm infer job finished, check actual output folder with `find`                |
| NaN loss on cluster                       | PyTorch 2.x numerical difference               | Check learning rate, batch size. Flag for discussion with Thomas.                 |
| GPU log empty                             | Job too short to capture entries               | Not a problem for long jobs -- ignore for validate job if only a few epochs       |
| Zone.Identifier files                     | Windows WinSCP transfer artifact               | `find $VSC_SCRATCH/dataset -name "*.Identifier" -delete`                          |
| find gives count of 0/1 on symlinked dirs | find does not follow symlinks by default       | Use `ls <symlink_path>                                                            | wc -l` instead of `find` for counting through symlinks |

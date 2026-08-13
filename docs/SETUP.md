# Environment Setup (Turing Cluster)

This documents the working environment setup on IIIT-H's Turing SLURM cluster,
including several non-obvious pitfalls hit during setup.

## Requesting a GPU node

```bash
ssh vedant.tejas@turing.iiit.ac.in
sinfo
srun --partition=u22 --account=priyesh.shukla --qos=high --pty bash
```

The login node (`turing`) has no GPU. `nvidia-smi` only works after `srun`
lands you on a compute node (e.g. `node01`, `node02`).

## CUDA toolkit

The system CUDA toolkit isn't on `PATH` by default:

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**Pitfall — do NOT add `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH`.**
`/usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs/libcuda.so` is a
link-time-only stub, not the real driver library. If it shadows the real
`libcuda.so` at `/usr/lib/x86_64-linux-gnu/`, PyTorch reports a misleading
"driver too old" error even though the driver is fine. Only put `cuda/bin`
on `PATH` (for `nvcc`/`nsys`/`ncu`); leave `LD_LIBRARY_PATH` alone.

## Python environment

```bash
python3 -m venv ~/torch-env
source ~/torch-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy flashinfer-bench modal tomli
```

**Pitfall — dependency conflicts can silently upgrade torch.**
Installing `flashinfer-bench`/`modal` pulled in `cuda-python 13.3.1`, which
depends on `cuda-bindings~=13.3.1`. Resolving that dependency chain silently
replaced `torch==2.11.0+cu128` with `torch==2.13.0+cu130` — a CUDA 13.0
build the installed driver (575.51.03, supports up to CUDA 12.9) can't run.
Symptom: `torch.cuda.is_available()` returns `False` with a "driver too old"
warning, even though `nvidia-smi` shows everything fine.

Fix: `pip uninstall torch -y && pip install torch --index-url
https://download.pytorch.org/whl/cu128` to force back to a cu128 build.
Recheck `torch.version.cuda` after any `pip install` that touches
`flashinfer-bench`/`modal`/`cuda-python`.

## Dataset (git-lfs)

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest ~/mlsys26-contest
export FIB_DATASET_PATH=~/mlsys26-contest
```

**Pitfall — `git-lfs` not installed means you get pointer files, not data.**
If `git-lfs` isn't on the system, `git clone` silently checks out ~130-byte
text pointer files instead of the real tensors. Everything *looks* cloned
(`du -sh` looks small but no error is raised) until something tries to parse
the "safetensors" file and fails with something like `Error while
deserializing header: header too large` (the parser is reading the pointer
text as if it were a binary header).

Fix (no-sudo, since `apt install git-lfs` may not be available):
```bash
cd ~
curl -L -o git-lfs.tar.gz https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar xzf git-lfs.tar.gz
export PATH=$HOME/git-lfs-3.5.1:$PATH
git lfs install --local
cd ~/mlsys26-contest && git lfs pull
```

## Verifying the environment end-to-end

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import flashinfer_bench; print('OK')"
which nsys ncu
```

GPU on this cluster: NVIDIA RTX 6000 Ada Generation, 48GB, driver 575.51.03,
CUDA 12.9. `ncu` (Nsight Compute) hardware performance counters were
confirmed accessible without special permissions on this cluster — no
`ERR_NVGPUCTRPERM` encountered.

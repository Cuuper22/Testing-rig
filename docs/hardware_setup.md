# Hardware Setup Guide

This project is designed to run on both GPU and TPU environments by leveraging
PyTorch Lightning or Hugging Face Accelerate. The sections below describe the
recommended hardware configurations and supporting software for common
accelerators.

## NVIDIA A100 GPUs

- **Driver & CUDA**: Install the latest NVIDIA drivers supported by your host OS
  and CUDA 11.4 or newer. Ensure that the CUDA toolkit version matches the
  version used to compile PyTorch.
- **PyTorch build**: Install a CUDA-enabled PyTorch build (e.g. `pip install
  torch --index-url https://download.pytorch.org/whl/cu118`).
- **Multi-GPU orchestration**: The provided `scripts/launch_training.sh` script
  reads `WORLD_SIZE`, `NODE_RANK`, and `NPROC_PER_NODE` to launch `torchrun`
  across multiple GPUs or nodes. When using PyTorch Lightning, set
  `trainer.accelerator=gpu` and `trainer.strategy=ddp` inside
  `configs/distributed.yaml`.
- **Networking**: For multi-node setups, configure a high-speed interconnect
  (InfiniBand or 100Gb Ethernet) and ensure that `MASTER_ADDR` and
  `MASTER_PORT` are reachable from all nodes.

## Google TPU v3 Pods

- **Environment**: Provision TPU v3 workers via Google Cloud or an on-premises
  TPU host. Install the latest libtpu package and TPU-enabled PyTorch build if
  training with PyTorch Lightning, or TPU-compatible XLA runtime when using
  Accelerate.
- **PyTorch Lightning**: Update `configs/distributed.yaml` to use
  `trainer.accelerator=tpu` and set `trainer.tpu_cores` to the desired number of
  cores (1, 8). Ensure that `PL_TORCH_DISTRIBUTED_BACKEND=xla` is exported when
  launching.
- **Hugging Face Accelerate**: Generate an Accelerate configuration with
  `accelerate config` specifying the TPU environment. The config file can be
  combined with the project YAML by setting `backend: accelerate`.
- **Cluster launch**: Use `scripts/launch_training.sh` with appropriate
  `WORLD_SIZE` and `NODE_RANK` values. When TPUs span multiple hosts, Google
  recommends using the TPU runtime's built-in launcher instead of `torchrun`.

## Common recommendations

- Install dependencies in a virtual environment or conda environment that is
  replicated across all nodes.
- Synchronize clocks between machines to avoid certificate validation issues
  for experiment tracking.
- Use a shared filesystem (NFS, GCSFuse, or S3 FUSE) for checkpoints when
  running distributed experiments.
- Validate connectivity with a short dry-run before starting long training
  jobs.

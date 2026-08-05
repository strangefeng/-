# Setup Environment

This document creates a clean YOLO smoke-test environment instead of continuing to repair the current Anaconda base environment.

## Recommended Environment

- Python: `3.10`
- NumPy: `1.26.4`
- OpenCV: `opencv-python==4.9.0.80`
- PyTorch: CUDA-enabled build installed from the `pytorch` and `nvidia` channels
- Ultralytics: latest stable release from pip

## Why a New Environment

- The current base environment is already in a mixed `conda` + `pip` state.
- `cv2` import is broken with `numpy._core.multiarray failed to import`.
- `ultralytics` depends on `cv2`, so YOLO training cannot be trusted in the current environment.

## Installation Steps

1. Open PowerShell in the repository root.
2. Run the setup script:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env_conda.ps1
```

## What Each Command Does

- `conda create -n yolo-smoke-py310 python=3.10 -y`
  Creates a fresh Python 3.10 environment.
- `conda activate yolo-smoke-py310`
  Activates the clean environment.
- `python -m pip install --upgrade pip`
  Updates pip inside the new environment.
- `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y`
  Installs a CUDA-enabled PyTorch stack suitable for NVIDIA GPUs like the local RTX 4060 Laptop GPU.
- `python -m pip install numpy==1.26.4 opencv-python==4.9.0.80 ultralytics pandas pyyaml`
  Installs the core smoke-test and training dependencies in a version combination that is typically stable for YOLO workflows.

## Optional Adjustment

- If GUI-enabled OpenCV is not needed, you may replace `opencv-python==4.9.0.80` with `opencv-python-headless==4.9.0.80`.

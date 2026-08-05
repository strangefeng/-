param(
    [string]$EnvName = 'yolo-smoke-py310'
)

$ErrorActionPreference = 'Stop'

$CondaHook = & conda shell.powershell hook
Invoke-Expression $CondaHook

conda create -n $EnvName python=3.10 -y
conda activate $EnvName

python -m pip install --upgrade pip
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install numpy==1.26.4 opencv-python==4.9.0.80 ultralytics pandas pyyaml

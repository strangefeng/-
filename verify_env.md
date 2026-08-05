# Verify Environment

Run these commands after activating the clean Conda environment.

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\.yolo-config"
& 'D:\yolo11\.conda\python.exe' -c "import numpy; print('numpy', numpy.__version__)"
& 'D:\yolo11\.conda\python.exe' -c "import cv2; print('cv2', cv2.__version__)"
& 'D:\yolo11\.conda\python.exe' -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
& 'D:\yolo11\.conda\python.exe' -c "import ultralytics; print('ultralytics', ultralytics.__version__)"
```

If all four commands complete successfully, the environment is ready for the smoke test.



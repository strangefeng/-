from __future__ import annotations

import os
from pathlib import Path


def configure_repo_runtime(repo_root: str | Path) -> Path:
    """Pin Ultralytics state inside the repo so imports do not depend on user-profile access."""
    root = Path(repo_root).resolve()
    config_dir = root / ".yolo-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    return config_dir

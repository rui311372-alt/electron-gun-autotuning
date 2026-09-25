from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = {
    "output_dir": "captured_tif/single.tif",
    "start_sleep_seconds": 5,
    "exposure_time_us": None,
}


@dataclass
class CameraCaptureConfig:
    output_dir: str = "../captured_tif/single.tif"
    start_sleep_seconds: int = 5
    exposure_time_us: int | None = None


def load_camera_config(config_path: Optional[Path] = None) -> tuple[CameraCaptureConfig, Path]:
    if config_path is None:
        config_path = ROOT / "camera_cap_config.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = CameraCaptureConfig(
            output_dir=data.get("output_dir", DEFAULT_CONFIG["output_dir"]),
            start_sleep_seconds=data.get("start_sleep_seconds", DEFAULT_CONFIG["start_sleep_seconds"]),
            exposure_time_us=(
                int(data["exposure_time_us"])
                if data.get("exposure_time_us") is not None
                else None
            ),
        )
    else:
        cfg = CameraCaptureConfig()

    return cfg, config_path

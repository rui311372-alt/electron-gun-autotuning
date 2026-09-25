from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def single_tif_path(cfg_output_dir: str) -> Path:
    return ROOT / cfg_output_dir

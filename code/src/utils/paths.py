# src/utils/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import pdb

pdb.set_trace()


def _ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Directories (created on module load)

DATA_DIR = _ensure_dir(PROJECT_ROOT / "data")
RESULTS_DIR = _ensure_dir(PROJECT_ROOT / "results")
CODE_DIR = _ensure_dir(PROJECT_ROOT / "code")
EXPERIMENTS_DIR = _ensure_dir(CODE_DIR / "experiments")
# SRC_DIR = _ensure_dir(CODE_DIR / "src")
#


def get_data_dir() -> Path:
    """Get absolute path to data directory."""
    return DATA_DIR


# TODO: change this, I think the origin is not samformer, but somewhere else
def get_samformer_dataset_path(dataset: str) -> Path:
    """Get absolute path to samformer dataset."""
    path = get_data_dir() / "samformer_datasets" / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Please ensure that you have downloaded the data via the scripts/download_samformer_dataset.py script"
        )
    return path


def get_data_path(filename: str, must_exist: bool = True) -> Path:
    """Get absolute path to a data file."""
    path = DATA_DIR / filename
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path


def get_results_path(experiment: str) -> Path:
    """Get absolute path to results folder (creates subdirectory)."""
    return _ensure_dir(RESULTS_DIR / experiment)


def get_experiment_output_path(experiment: str, filename: str) -> Path:
    """Get path for saving experiment output."""
    return get_results_path(experiment) / filename

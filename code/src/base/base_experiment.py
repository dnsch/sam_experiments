import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import argparse

import numpy as np
import torch


class BaseExperiment(ABC):
    """
    Base class for all experiment types.

    Provides common infrastructure:
        - Script directory detection
        - Path setup
        - Configuration parsing
        - Logging setup
        - Seed management

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - get_path_config(): Return path configuration dict
        - get_log_dir_components(args): Return log directory path components
        - setup_dataloader(args, logger): Setup and return dataloader
        - run(): Main entry point

    """

    def __init__(self):
        self.script_dir = self._get_script_dir()
        self._setup_paths()

        # Common state - will be set during setup
        self.args: Optional[argparse.Namespace] = None
        self.log_dir: Optional[Path] = None
        self.logger = None

    # =========================================================================
    # Path Infrastructure
    # =========================================================================

    def _get_script_dir(self) -> Path:
        """Get the directory of the calling script."""
        import inspect

        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                return Path(frame_info.filename).resolve().parent
        return Path(__file__).resolve().parent

    @abstractmethod
    def get_path_config(self) -> Dict[str, Any]:
        """
        Return path configuration.

        Expected keys:
            - 'sys_path_parents': List[int] - parent levels to add to sys.path
            - 'results_parent': int - parent level for results directory
            - 'results_subdir': str - subdirectory under results/

        """
        pass

    def _setup_paths(self):
        """Setup sys.path entries based on path config."""
        config = self.get_path_config()
        for level in config.get("sys_path_parents", []):
            path = self.script_dir.parents[level]
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)

    def _get_base_results_dir(self) -> Path:
        """Get the base results directory."""
        config = self.get_path_config()
        return (
            self.script_dir.parents[config["results_parent"]] / "results" / config["results_subdir"]
        )

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def get_config_parser(self) -> argparse.ArgumentParser:
        """Return the argument parser for this experiment."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name string."""
        pass

    @abstractmethod
    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """
        Return tuple of path components for log directory.

        Example: ('ModelName', 'dataset_name', 'seq_len_96_pred_len_24')
        """
        pass

    @abstractmethod
    def setup_dataloader(self, args: argparse.Namespace, logger=None):
        """Setup and return dataloader(s)."""
        pass

    @abstractmethod
    def run(self):
        """Main entry point for the experiment."""
        pass

    # =========================================================================
    # Common Implementation
    # =========================================================================

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    def get_config(self) -> Tuple[argparse.Namespace, Path, Any]:
        """Setup configuration, logging directory, and logger."""
        from src.utils.logging import get_logger

        parser = self.get_config_parser()
        args = parser.parse_args()
        args.model_name = self.get_model_name()

        # Store parser reference if needed
        args._parser = parser

        # Build log directory path
        base_dir = self._get_base_results_dir()
        components = self.get_log_dir_components(args)

        log_dir = base_dir.joinpath(*components)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = get_logger(log_dir, __name__, f"record_s{args.seed}.log")
        logger.info(args)

        # Store in instance
        self.args = args
        self.log_dir = log_dir
        self.logger = logger

        return args, log_dir, logger

    def _setup_loss_function(self, args: argparse.Namespace):
        """Setup loss function from args if loss_name is specified."""
        if hasattr(args, "loss_name"):
            from src.utils.loss_functions import get_loss_function

            args.loss_fn = get_loss_function(args.loss_name)
        return args


def run_experiment(experiment_class: type):
    """
    Generic runner for any experiment class.
    Use in if __name__ == "__main__" blocks.
    """
    experiment = experiment_class()
    return experiment.run()

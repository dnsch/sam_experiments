import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List
import argparse

import numpy as np
import torch


class NixtlaSequentialComparison(ABC):
    """
    Base class for statsforecast-based training scripts.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_statsforecast_model(args): Create and return the SF model

    """

    def __init__(self):
        self.script_dir = self._get_script_dir()
        self._setup_paths()

    def _get_script_dir(self) -> Path:
        import inspect

        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                return Path(frame_info.filename).resolve().parent
        return Path(__file__).resolve().parent

    def _setup_paths(self):
        """Setup common sys.path entries for nixtla models."""
        base_paths = [
            self.script_dir.parents[4],
            self.script_dir.parents[5],
        ]
        for path in base_paths:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)

    def _get_base_results_dir(self) -> Path:
        return self.script_dir.parents[5] / "results" / "sequential_comparison"

    # =========================================================================
    # Abstract methods
    # =========================================================================

    @abstractmethod
    def get_config_parser(self) -> argparse.ArgumentParser:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def create_statsforecast_model(self, args: argparse.Namespace):
        """Create and return the statsforecast model instance."""
        pass

    # =========================================================================
    # Optional overrides
    # =========================================================================

    def get_sf_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for StatsForecast initialization."""
        return {
            "freq": args.freq,
            "n_jobs": getattr(args, "n_jobs", -1),
        }

    def get_engine_kwargs(self, args, sf, data, logger, log_dir):
        """Override to add AutoARIMA-specific kwargs."""
        from src.utils.loss_functions import get_loss_function

        return {
            "model": sf,
            "dataloader": data,
            "scaler": None,
            "pred_len": args.horizon,
            "loss_fn": get_loss_function(args.loss_name),
            "backend": "statsforecast",
            "num_channels": data[0]["unique_id"].nunique(),
            "freq": getattr(args, "freq", "H"),  # Default hourly
            "n_jobs": getattr(args, "n_jobs", -1),
            "logger": logger,
            "log_dir": log_dir,
            "seed": args.seed,
            "enable_plotting": getattr(args, "enable_plotting", True),
            "metrics": getattr(args, "metrics", None),
            "args": args,
        }

    # =========================================================================
    # Core implementation
    # =========================================================================

    def set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    def get_config(self):
        from src.utils.logging import get_logger
        from src.utils.loss_functions import get_loss_function

        parser = self.get_config_parser()
        args = parser.parse_args()
        args.model_name = self.get_model_name()
        import pdb

        pdb.set_trace()
        args.loss_fn = get_loss_function(args.loss_name)

        base_dir = self._get_base_results_dir()
        log_dir = (
            base_dir
            / args.model_name
            / args.dataset
            / f"seq_len_{args.seq_len}_pred_len_{args.horizon}"
        )
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(log_dir, __name__, f"record_s{args.seed}.log")
        logger.info(args)

        return args, log_dir, logger

    def setup_dataloader(self, args, logger):
        from src.utils.dataloader import StatsforecastDataloader

        dataloader_instance = StatsforecastDataloader(
            dataset=args.dataset,
            args=args,
            logger=logger,
            merge_train_val=True,
        )
        return dataloader_instance.get_dataloader()

    def run_experiments(self, data_list, args, logger, log_dir) -> List:
        """Execute training for each data entry."""
        from statsforecast import StatsForecast
        from src.base.nixtla_engine import NixtlaEngine

        results = []

        for idx, data in enumerate(data_list):
            print(f"\n{'=' * 60}")
            print(f"Processing Experiment {idx + 1}/{len(data_list)}")
            print(f"{'=' * 60}\n")

            # Create model
            sf_model = self.create_statsforecast_model(args)

            # Create StatsForecast instance
            sf_kwargs = self.get_sf_kwargs(args)
            sf = StatsForecast(models=[sf_model], **sf_kwargs)

            # Create experiment log dir
            experiment_log_dir = log_dir / f"experiment_{idx}"

            # Create engine
            engine_kwargs = self.get_engine_kwargs(args, sf, data, logger, experiment_log_dir)
            engine = NixtlaEngine(**engine_kwargs)

            # Train
            result = engine.train()
            results.append(result)

            print(f"\nCompleted Experiment {idx + 1}/{len(data_list)}\n")

        return results

    def run(self):
        """Main entry point."""
        args, log_dir, logger = self.get_config()
        self.set_seed(args.seed)

        data = self.setup_dataloader(args, logger)
        results = self.run_experiments(data, args, logger, log_dir)

        # Analyze results
        for idx, result in enumerate(results):
            print(f"Experiment {idx}: {result}")

        return results


def run_sequential_comparison(training_class: type):
    trainer = training_class()
    trainer.run()

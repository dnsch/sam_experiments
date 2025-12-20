from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse

from src.base.base_experiment import BaseExperiment


class NixtlaSequentialComparison(BaseExperiment):
    """
    Base class for statsforecast-based sequential comparison experiments.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_statsforecast_model(args): Create and return the SF model

    Optional overrides:
        - get_sf_kwargs(args): Custom StatsForecast initialization kwargs
        - get_engine_kwargs(...): Custom NixtlaEngine kwargs

    """

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self) -> str:
        """Results go under results/sequential_comparison/"""
        return "sequential_comparison"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components."""
        return (
            args.model_name,
            args.dataset,
            f"seq_len_{args.seq_len}_pred_len_{args.horizon}",
        )

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def create_statsforecast_model(self, args: argparse.Namespace):
        """Create and return the statsforecast model instance."""
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def get_sf_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for StatsForecast initialization."""
        return {
            "freq": args.freq,
            "n_jobs": getattr(args, "n_jobs", -1),
        }

    def get_engine_kwargs(
        self,
        args: argparse.Namespace,
        sf,
        data,
        logger,
        log_dir: Path,
    ) -> Dict[str, Any]:
        """Get kwargs for NixtlaEngine initialization."""
        from src.utils.loss_functions import get_loss_function

        return {
            "model": sf,
            "dataloader": data,
            "scaler": None,
            "pred_len": args.horizon,
            "loss_fn": get_loss_function(args.loss_name),
            "backend": "statsforecast",
            "num_channels": data[0]["unique_id"].nunique(),
            "freq": getattr(args, "freq", "H"),
            "n_jobs": getattr(args, "n_jobs", -1),
            "logger": logger,
            "log_dir": log_dir,
            "seed": args.seed,
            "enable_plotting": getattr(args, "enable_plotting", True),
            "metrics": getattr(args, "metrics", None),
            "args": args,
        }

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def setup_dataloader(self, args: argparse.Namespace, logger=None):
        """Setup statsforecast dataloader."""
        from src.utils.dataloader import StatsforecastDataloader

        dataloader_instance = StatsforecastDataloader(
            dataset=args.dataset,
            args=args,
            logger=logger,
            merge_train_val=True,
        )
        return dataloader_instance.get_dataloader()

    def run_experiments(
        self,
        data_list: List,
        args: argparse.Namespace,
        logger,
        log_dir: Path,
    ) -> List:
        """Execute training for each data entry."""
        from statsforecast import StatsForecast
        from src.base.nixtla_engine import NixtlaEngine

        results = []

        for idx, data in enumerate(data_list):
            self._log_experiment_start(idx, len(data_list))

            # Create model and StatsForecast instance
            sf_model = self.create_statsforecast_model(args)
            sf = StatsForecast(models=[sf_model], **self.get_sf_kwargs(args))

            # Create experiment log dir
            experiment_log_dir = log_dir / f"experiment_{idx}"

            # Create and run engine
            engine_kwargs = self.get_engine_kwargs(args, sf, data, logger, experiment_log_dir)
            engine = NixtlaEngine(**engine_kwargs)
            result = engine.train()
            results.append(result)

            self._log_experiment_end(idx, len(data_list))

        return results

    def _log_experiment_start(self, idx: int, total: int):
        """Log experiment start."""
        print(f"\n{'=' * 60}")
        print(f"Processing Experiment {idx + 1}/{total}")
        print(f"{'=' * 60}\n")

    def _log_experiment_end(self, idx: int, total: int):
        """Log experiment end."""
        print(f"\nCompleted Experiment {idx + 1}/{total}\n")

    def run(self):
        """Main entry point."""
        args, log_dir, logger = self.get_config()
        self._setup_loss_function(args)
        self.set_seed(args.seed)

        data = self.setup_dataloader(args, logger)
        results = self.run_experiments(data, args, logger, log_dir)

        # Log results summary
        for idx, result in enumerate(results):
            logger.info(f"Experiment {idx}: {result}")

        return results


def run_sequential_comparison(training_class: type):
    """Run a sequential comparison experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(training_class)

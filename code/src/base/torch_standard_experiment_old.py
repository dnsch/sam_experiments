import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.set_num_threads(3)


from src.utils.paths import get_data_path, get_results_path, get_experiment_output_path
import src.utils.paths


class TorchStandardExperiment(ABC):
    """
    Base class for torch standard experiment scripts.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_model(args, dataloader): Create and return model
        - get_engine_class(): Return the engine class to use

    Optional overrides:
        - get_model_init_kwargs(args, dataloader): Custom model kwargs
        - get_engine_init_kwargs(...): Custom engine kwargs
        - get_log_dir_name(args): Custom log directory naming
        - post_training_hooks(args, model, dataloader, log_dir, logger):

          Custom post-training analysis
    """

    def __init__(self):
        self.script_dir = self._get_script_dir()
        self._setup_paths()

        # Will be set during setup
        self.args: Optional[argparse.Namespace] = None
        self.log_dir: Optional[Path] = None
        self.logger = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader_instance = None
        self.dataloader: Optional[Dict] = None
        self.engine = None
        self.scaler = None

    def _get_script_dir(self) -> Path:
        """Get the directory of the calling script. Override if needed."""
        import inspect

        # Get the file path of the class that's actually being instantiated
        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                return Path(frame_info.filename).resolve().parent
        return Path(__file__).resolve().parent

    def _setup_paths(self):
        """Setup common sys.path entries."""
        # Common paths - adjust based on your project structure
        base_paths = [
            self.script_dir.parents[1],  # experiments
            self.script_dir.parents[2],  # code
        ]

        # Optional utility paths
        optional_paths = [
            self.script_dir.parents[2] / "lib" / "utils" / "pyhessian",
            self.script_dir.parents[2] / "lib" / "utils" / "loss_landscape",
        ]

        for path in base_paths + optional_paths:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)

    def _get_base_results_dir(self) -> Path:
        """Get the base results directory."""
        return self.script_dir.parents[3] / "results" / "standard"

    # These have to be implemented in custom training scripts
    @abstractmethod
    def get_config_parser(self) -> argparse.ArgumentParser:
        """Return the argument parser for this model."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name string (e.g., 'autoformer', 'patchtst')."""
        pass

    @abstractmethod
    def create_model(
        self, args: argparse.Namespace, dataloader: Dict
    ) -> torch.nn.Module:
        """Create and return the model instance."""
        pass

    @abstractmethod
    def get_engine_class(self):
        """Return the engine class to use for training."""
        pass

    # Optional override methods
    def get_log_dir_suffix(self, args: argparse.Namespace) -> str:
        """
        Get the log directory suffix based on optimizer type.
        Override for custom naming.
        """
        model_name = self.get_model_name()

        if getattr(args, "sam", False):
            return f"{model_name}SAM"
        elif getattr(args, "gsam", False):
            return f"{model_name}GSAM"
        else:
            return model_name

    # TODO: create string dynamically, based on used args
    def get_log_dir_params(self, args: argparse.Namespace) -> str:
        """
        Get parameter-specific part of log directory path.
        Override for custom parameter naming.
        """
        base_params = (
            f"seq_len_{args.seq_len}_pred_len_{args.horizon}_bs_{args.batch_size}"
        )

        if getattr(args, "sam", False):
            return f"{base_params}_rho_{args.rho}"
        elif getattr(args, "gsam", False):
            return f"{base_params}_gsam_alpha_{args.gsam_alpha}_rho_max_{args.gsam_rho_max}"
        else:
            return base_params

    def get_dataloader_class(self):
        """Return the dataloader class to use. Override for custom dataloaders."""
        from src.utils.dataloader import SamformerDataloader

        return SamformerDataloader

    # TODO: put sequential_comparison in its own base experiment file
    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization. Override for custom args."""
        return {
            "dataset": args.dataset,
            "seq_len": args.seq_len,
            "pred_len": args.horizon,
            "seed": args.seed,
            "time_increment": getattr(args, "time_increment", 1),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "batch_size": args.batch_size,
            "sequential_comparison": getattr(args, "sequential_comparison", False),
        }

    def get_loss_function(self) -> torch.nn.Module:
        """Get the loss function. Override for custom loss."""
        return torch.nn.MSELoss()

    def get_metrics(self) -> List[str]:
        """Get metrics to track. Override for custom metrics."""
        return ["mse"]

    def get_engine_kwargs(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        scaler,
        optimizer,
        scheduler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ) -> Dict[str, Any]:
        """
        Get kwargs for engine initialization.
        Override to add model-specific engine parameters.
        """
        return {
            "device": args.device,
            "model": model,
            "dataloader": dataloader,
            "scaler": scaler,
            "loss_fn": loss_fn,
            "lrate": args.lrate,
            "optimizer": optimizer,
            "sam": getattr(args, "sam", False),
            "gsam": getattr(args, "gsam", False),
            "scheduler": scheduler,
            "clip_grad_value": args.clip_grad_value,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "log_dir": log_dir,
            "logger": logger,
            "seed": args.seed,
            "metrics": self.get_metrics(),
        }

    def post_training_hooks(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        log_dir: Path,
        logger,
        loss_fn: torch.nn.Module,
    ):
        """
        Run after training/testing completes. Override for custom post-training analysis.
        Default implementation handles Hessian analysis if enabled.
        """
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(
                args, model, dataloader, log_dir, logger, loss_fn
            )

    # Core methods
    def get_config(self) -> Tuple[argparse.Namespace, Path, Any]:
        """Setup configuration, logging directory, and logger."""
        from src.utils.logging import get_logger

        parser = self.get_config_parser()
        args = parser.parse_args()
        args._parser = parser
        args.model_name = self.get_model_name()

        # Build log directory path
        base_dir = self._get_base_results_dir()
        log_dir_suffix = self.get_log_dir_suffix(args)
        log_dir_params = self.get_log_dir_params(args)

        log_dir = base_dir / log_dir_suffix / args.dataset / log_dir_params
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = get_logger(log_dir, __name__, f"record_s{args.seed}.log")
        logger.info(args)

        return args, log_dir, logger

    def setup_dataloader(self, args: argparse.Namespace) -> Tuple[Any, Dict]:
        """Initialize and return dataloader instance and dataloader dict."""
        DataloaderClass = self.get_dataloader_class()
        kwargs = self.get_dataloader_kwargs(args)

        dataloader_instance = DataloaderClass(**kwargs)
        dataloader = dataloader_instance.get_dataloader()

        return dataloader_instance, dataloader

    def setup_optimizer(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        logger,
    ) -> torch.optim.Optimizer:
        """Setup and return the optimizer."""
        from src.utils.model_utils import load_optimizer

        return load_optimizer(model, args, logger)

    def setup_optimizer_and_scheduler(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        dataloader: Dict,
        logger,
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Setup optimizer and learning rate scheduler with SAM/GSAM support.
        Returns (optimizer, scheduler) tuple.
        """
        from src.utils.samformer_utils.sam import SAM
        from lib.optimizers.gsam.gsam.gsam import GSAM
        from lib.optimizers.gsam.gsam.scheduler import LinearScheduler

        # Base optimizer
        optimizer = self.setup_optimizer(model, args, logger)
        lr_scheduler = None

        if getattr(args, "sam", False):
            # SAM optimizer
            base_optimizer_class = getattr(torch.optim, args.optimizer)
            optimizer = SAM(
                params=model.parameters(),
                base_optimizer=base_optimizer_class,
                rho=args.rho,
                lr=args.lrate,
                weight_decay=args.wdecay,
            )
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=5,
                T_mult=1,
                eta_min=1e-6,
                last_epoch=-1,
            )

        elif getattr(args, "gsam", False):
            # GSAM optimizer
            lr_scheduler = LinearScheduler(
                T_max=args.max_epochs * len(dataloader["train_loader"]),
                max_value=args.lrate,
                min_value=args.lrate * 0.01,
                optimizer=optimizer,
            )
            rho_scheduler = LinearScheduler(
                T_max=args.max_epochs * len(dataloader["train_loader"]),
                max_value=args.gsam_rho_max,
                min_value=args.gsam_rho_min,
            )
            optimizer = GSAM(
                params=model.parameters(),
                base_optimizer=optimizer,
                model=model,
                gsam_alpha=args.gsam_alpha,
                rho_scheduler=rho_scheduler,
                adaptive=args.gsam_adaptive,
            )
        else:
            # Standard scheduler
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=5,
                T_mult=1,
                eta_min=1e-6,
                last_epoch=-1,
            )

        return optimizer, lr_scheduler

    def create_engine(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        scaler,
        optimizer,
        scheduler,
        loss_fn: torch.nn.Module,
        log_dir: Path,
        logger,
    ):
        """Create and return the training engine."""
        EngineClass = self.get_engine_class()
        kwargs = self.get_engine_kwargs(
            args,
            model,
            dataloader,
            scaler,
            optimizer,
            scheduler,
            loss_fn,
            log_dir,
            logger,
        )
        return EngineClass(**kwargs)

    def _run_hessian_analysis(
        self,
        args: argparse.Namespace,
        model: torch.nn.Module,
        dataloader: Dict,
        log_dir: Path,
        logger,
        loss_fn: torch.nn.Module,
    ):
        """Run Hessian analysis if enabled."""
        from lib.utils.pyhessian.pyhessian import hessian
        from lib.utils.pyhessian.density_plot import get_esd_plot

        logger.info("Computing Hessian analysis...")

        hessian_comp = hessian(
            model, loss_fn, dataloader=dataloader["train_loader"], cuda=args.device
        )
        density_eigen, density_weight = hessian_comp.density()
        get_esd_plot(density_eigen, density_weight)

        if getattr(args, "hessian_directions", False):
            from src.utils.hessian_utils import (
                compute_dominant_hessian_directions,
                save_eigenvectors_to_hdf5,
            )

            max_ev, max_evec, min_ev, min_evec = compute_dominant_hessian_directions(
                model,
                loss_fn,
                dataloader["train_loader"],
                tol=1e-4,
            )
            save_eigenvectors_to_hdf5(
                args=args,
                net=model,
                max_evec=max_evec,
                min_evec=min_evec,
                output_dir=log_dir / "hessian_directions",
            )
            logger.info("Hessian directions saved.")

    def run(self):
        """
        Runs the complete training pipeline.
        """
        from src.utils.reproducibility import set_seed

        # Setup configuration
        self.args, self.log_dir, self.logger = self.get_config()

        # Set seed for reproducibility
        set_seed(self.args.seed)

        # Setup dataloader
        self.dataloader_instance, self.dataloader = self.setup_dataloader(self.args)
        self.scaler = self.dataloader_instance.get_scaler()

        # Create model
        self.model = self.create_model(self.args, self.dataloader)
        self.model.print_model_summary(self.args, self.logger)

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler(
            self.model, self.args, self.dataloader, self.logger
        )

        # Get loss function
        loss_fn = self.get_loss_function()

        # Create engine
        self.engine = self.create_engine(
            self.args,
            self.model,
            self.dataloader,
            self.scaler,
            self.optimizer,
            self.scheduler,
            loss_fn,
            self.log_dir,
            self.logger,
        )

        # Train or test
        if self.args.mode == "train":
            result = self.engine.train()
            self.logger.info(f"Training completed. Result: {result}")
        elif self.args.mode == "test":
            result = self.engine.evaluate(self.args.mode)
            self.logger.info(f"Testing completed. Result: {result}")

        # Post-training hooks (e.g., Hessian analysis)
        self.post_training_hooks(
            self.args, self.model, self.dataloader, self.log_dir, self.logger, loss_fn
        )

        return result


def run_standard_experiment(experiment_class: type):
    """
    Run a training class.
    Use in if __name__ == "__main__" blocks.
    """
    trainer = experiment_class()
    trainer.run()

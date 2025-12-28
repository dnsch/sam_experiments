from abc import abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.set_num_threads(3)

from src.base.base_experiment import BaseExperiment


class TorchStandardExperiment(BaseExperiment):
    """
    Base class for PyTorch standard training experiments.

    Subclasses must implement:
        - get_config_parser(): Return argparse parser
        - get_model_name(): Return model name string
        - create_model(args, dataloader): Create and return model
        - get_engine_class(): Return the engine class to use

    Optional overrides:
        - get_log_dir_suffix(args): Custom model name suffix
        - get_log_dir_params(args): Custom parameter string
        - get_dataloader_class(): Custom dataloader class
        - get_dataloader_kwargs(args): Custom dataloader kwargs
        - get_loss_function(): Custom loss function
        - get_metrics(): Custom metrics list
        - get_engine_kwargs(...): Custom engine kwargs
        - post_training_hooks(...): Custom post-training analysis

    """

    def __init__(self):
        super().__init__()

        # Torch-specific state
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.dataloader_instance = None
        self.dataloader: Optional[Dict] = None
        self.engine = None
        self.scaler = None

    # =========================================================================
    # Path Configuration (implements BaseExperiment abstract methods)
    # =========================================================================

    def get_results_subdir(self) -> str:
        """Results go under results/standard/"""
        return "standard"

    def get_log_dir_components(self, args: argparse.Namespace) -> Tuple[str, ...]:
        """Return log directory path components."""
        return (
            self.get_log_dir_suffix(args),
            args.dataset,
            self.get_log_dir_params(args),
        )

    def get_log_dir_suffix(self, args: argparse.Namespace) -> str:
        """Get model name suffix based on optimizer type."""
        model_name = self.get_model_name()
        if getattr(args, "sam", False):
            return f"{model_name}SAM"
        elif getattr(args, "gsam", False):
            return f"{model_name}GSAM"
        return model_name

    def get_log_dir_params(self, args: argparse.Namespace) -> str:
        """Get parameter-specific part of log directory path."""
        base_params = f"seq_len_{args.seq_len}_pred_len_{args.horizon}_bs_{args.batch_size}"
        if getattr(args, "sam", False):
            return f"{base_params}_rho_{args.rho}"
        elif getattr(args, "gsam", False):
            return f"{base_params}_gsam_alpha_{args.gsam_alpha}_rho_max_{args.gsam_rho_max}"
        return base_params

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def create_model(self, args: argparse.Namespace, dataloader: Dict) -> torch.nn.Module:
        """Create and return the model instance."""
        pass

    @abstractmethod
    def get_engine_class(self):
        """Return the engine class to use for training."""
        pass

    # =========================================================================
    # Optional Overrides
    # =========================================================================

    def get_dataloader_class(self):
        """Return the dataloader class to use."""
        from src.utils.dataloader import SamformerDataloader

        return SamformerDataloader

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization."""
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
        """Get the loss function."""
        return torch.nn.MSELoss()

    def get_metrics(self) -> List[str]:
        """Get metrics to track."""
        return ["mse"]

    # TODO: change every argument to num_channels or let it be consistent in
    # order to avoid this function
    def get_revin_num_features(self, args: argparse.Namespace) -> Optional[int]:
        """
        Get the number of features for RevIN.

        Override in subclasses to return the appropriate number of channels/features.
        Common options:
            - args.num_channels (SAMFormer, TSMixer)
            - args.enc_in (Transformers, PatchTST, DLinear)

        If None is returned, the engine will try to get it from model.num_channels.

        Args:
            args: Parsed arguments

        Returns:
            Number of features for RevIN, or None to auto-detect from model
        """
        # Try common argument names in order of preference
        if hasattr(args, "num_channels"):
            return args.num_channels
        if hasattr(args, "enc_in"):
            return args.enc_in
        # Return None to let engine auto-detect from model
        return None

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
        """Get kwargs for engine initialization."""
        return {
            "device": args.device,
            "model": model,
            "dataloader": dataloader,
            "scaler": scaler,
            "loss_fn": loss_fn,
            "lrate": args.lrate,
            "optimizer": optimizer,
            # RevIN parameters
            "use_revin": getattr(args, "use_revin", False),
            "revin_affine": getattr(args, "revin_affine", False),
            "revin_num_features": self.get_revin_num_features(args),
            "revin_subtract_last": getattr(args, "revin_subtract_last", False),
            # Sharpness Aware Minimization
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
        """Run after training/testing completes."""
        if getattr(args, "hessian_analysis", False):
            self._run_hessian_analysis(args, model, dataloader, log_dir, logger, loss_fn)

    # =========================================================================
    # Core Implementation
    # =========================================================================

    def setup_dataloader(self, args: argparse.Namespace, logger=None) -> Tuple[Any, Dict]:
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
        """Setup optimizer and learning rate scheduler with SAM/GSAM support."""
        optimizer = self.setup_optimizer(model, args, logger)
        lr_scheduler = None

        if getattr(args, "sam", False):
            optimizer, lr_scheduler = self._setup_sam_optimizer(model, args)
        elif getattr(args, "gsam", False):
            optimizer, lr_scheduler = self._setup_gsam_optimizer(model, args, dataloader, optimizer)
        else:
            lr_scheduler = self._setup_standard_scheduler(optimizer)

        return optimizer, lr_scheduler

    def _setup_sam_optimizer(self, model: torch.nn.Module, args: argparse.Namespace) -> Tuple:
        """Setup SAM optimizer."""
        from src.utils.samformer_utils.sam import SAM

        base_optimizer_class = getattr(torch.optim, args.optimizer)
        optimizer = SAM(
            params=model.parameters(),
            base_optimizer=base_optimizer_class,
            rho=args.rho,
            lr=args.lrate,
            weight_decay=args.wdecay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
        )
        return optimizer, scheduler

    def _setup_gsam_optimizer(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        dataloader: Dict,
        base_optimizer: torch.optim.Optimizer,
    ) -> Tuple:
        """Setup GSAM optimizer."""
        from lib.optimizers.gsam.gsam.gsam import GSAM
        from lib.optimizers.gsam.gsam.scheduler import LinearScheduler

        T_max = args.max_epochs * len(dataloader["train_loader"])

        lr_scheduler = LinearScheduler(
            T_max=T_max,
            max_value=args.lrate,
            min_value=args.lrate * 0.01,
            optimizer=base_optimizer,
        )
        rho_scheduler = LinearScheduler(
            T_max=T_max,
            max_value=args.gsam_rho_max,
            min_value=args.gsam_rho_min,
        )
        optimizer = GSAM(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            gsam_alpha=args.gsam_alpha,
            rho_scheduler=rho_scheduler,
            adaptive=args.gsam_adaptive,
        )
        return optimizer, lr_scheduler

    def _setup_standard_scheduler(self, optimizer: torch.optim.Optimizer):
        """Setup standard learning rate scheduler."""
        return CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
        )

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
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
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
                model, loss_fn, dataloader["train_loader"], tol=1e-4
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
        """Runs the complete training pipeline."""
        from src.utils.reproducibility import set_seed

        # Setup configuration
        args, log_dir, logger = self.get_config()

        # Set seed for reproducibility
        set_seed(args.seed)

        # Setup dataloader
        self.dataloader_instance, self.dataloader = self.setup_dataloader(args)
        self.scaler = self.dataloader_instance.get_scaler()

        # Create model
        self.model = self.create_model(args, self.dataloader)
        self.model.print_model_summary(args, logger)

        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self.setup_optimizer_and_scheduler(
            self.model, args, self.dataloader, logger
        )

        # Get loss function and create engine
        loss_fn = self.get_loss_function()
        self.engine = self.create_engine(
            args,
            self.model,
            self.dataloader,
            self.scaler,
            self.optimizer,
            self.scheduler,
            loss_fn,
            log_dir,
            logger,
        )

        # Train or test
        if args.mode == "train":
            result = self.engine.train()
            logger.info(f"Training completed. Result: {result}")
        elif args.mode == "test":
            result = self.engine.evaluate(args.mode)
            logger.info(f"Testing completed. Result: {result}")

        # Post-training hooks
        self.post_training_hooks(args, self.model, self.dataloader, log_dir, logger, loss_fn)

        return result


def run_standard_experiment(experiment_class: type):
    """Run a standard torch experiment."""
    from src.base.base_experiment import run_experiment

    return run_experiment(experiment_class)

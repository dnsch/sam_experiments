from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.samformer import SAMFormer
from src.engines.samformer_engine import SAMFormer_Engine
from src.utils.args import get_samformer_config


class SAMFormerExperiment(TorchStandardExperiment):
    """SAMFormer-specific training implementation."""

    def get_config_parser(self):
        return get_samformer_config()

    def get_model_name(self):
        return "samformer"

    def get_engine_class(self):
        return SAMFormer_Engine

    def get_log_dir_suffix(self, args):
        """Override to use 'simple_transformer' when SAM is disabled."""
        if getattr(args, "sam", False):
            return "samformer"
        elif getattr(args, "gsam", False):
            return "samformerGSAM"
        return "simple_transformer"

    def get_metrics(self):
        """Override to specify SAMFormer metrics."""
        return ["mse", "mape", "rmse"]

    def get_engine_kwargs(
        self, args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
    ):
        """Override to add SAMFormer-specific engine parameters."""
        kwargs = super().get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )
        kwargs["plot_attention"] = getattr(args, "plot_attention", True)
        return kwargs

    def create_model(self, args, dataloader):
        # Get num_channels from dataloader
        num_channels = dataloader["train_loader"].dataset[0][0].shape[0]

        return SAMFormer(
            num_channels=num_channels,
            seq_len=args.seq_len,
            hid_dim=args.hid_dim,
            horizon=args.horizon,
            plot_attention=getattr(args, "plot_attention", True),
        )

    def post_training_hooks(self, args, model, dataloader, log_dir, logger, loss_fn):
        """Run Hessian analysis after training if enabled."""
        # Call parent's Hessian analysis if enabled
        super().post_training_hooks(args, model, dataloader, log_dir, logger, loss_fn)


if __name__ == "__main__":
    run_standard_experiment(SAMFormerExperiment)

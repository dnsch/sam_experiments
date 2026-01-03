from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)

# from src.models.time_series.tsmixer.tsmixer import TSMixer
from src.models.time_series.tsmixer import TSMixer
from src.engines.tsmixer_engine import TSMixer_Engine
from src.utils.args import get_tsmixer_config


class TSMixerExperiment(TorchStandardExperiment):
    """TSMixer-specific training implementation."""

    def get_config_parser(self):
        return get_tsmixer_config()

    def get_model_name(self):
        return "tsmixer"

    def get_engine_class(self):
        return TSMixer_Engine

    def get_metrics(self):
        """Override to specify TSMixer metrics."""
        return ["mape", "rmse"]

    def get_engine_kwargs(
        self, args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
    ):
        """Override to add TSMixer-specific engine parameters."""
        kwargs = super().get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )

        num_channels = dataloader["train_loader"].dataset[0][0].shape[0]

        kwargs.update(
            {
                "batch_size": args.batch_size,
                "num_channels": num_channels,
                "pred_len": args.pred_len,
                "use_revin": getattr(args, "use_revin", False),
            }
        )
        return kwargs

    def create_model(self, args, dataloader):
        """Create TSMixer model instance."""
        num_channels = dataloader["train_loader"].dataset[0][0].shape[0]

        return TSMixer(
            num_channels=num_channels,
            input_dim=num_channels,
            output_dim=num_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            activation_fn=args.activation_fn,
            num_blocks=args.num_blocks,
            dropout_rate=args.dropout_rate,
            ff_dim=args.ff_dim,
            normalize_before=args.normalize_before,
            norm_type=args.norm_type,
        )


if __name__ == "__main__":
    run_standard_experiment(TSMixerExperiment)

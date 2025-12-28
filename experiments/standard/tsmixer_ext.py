from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)

# from src.models.time_series.tsmixer.tsmixer_ext import TSMixerExt
from src.models.time_series.tsmixer import TSMixerExt
from src.engines.tsmixer_ext_engine import TSMixerExt_Engine
from src.utils.args import get_tsmixer_ext_config


class TSMixerExtExperiment(TorchStandardExperiment):
    """TSMixerExt-specific training implementation."""

    def get_config_parser(self):
        return get_tsmixer_ext_config()

    def get_model_name(self):
        return "tsmixer_ext"

    def get_engine_class(self):
        return TSMixerExt_Engine

    def get_metrics(self):
        return ["mape", "rmse"]

    def get_dataloader_kwargs(self, args):
        """Override to enable time features for TSMixerExt."""
        kwargs = super().get_dataloader_kwargs(args)
        kwargs.update(
            {
                "model_type": "tsmixer_ext",
                "use_time_features": True,
                "num_static_features": args.static_channels,
            }
        )
        return kwargs

    def get_engine_kwargs(
        self, args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
    ):
        kwargs = super().get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )

        # TSMixerExtDataset returns (seq_len, channels), so use shape[-1]
        sample = dataloader["train_loader"].dataset[0]
        num_channels = sample[0].shape[-1]
        extra_channels = sample[1].shape[-1]
        static_channels = sample[3].shape[0]

        kwargs.update(
            {
                "batch_size": args.batch_size,
                "num_channels": num_channels,
                "pred_len": args.horizon,
                "extra_channels": extra_channels,
                "static_channels": static_channels,
            }
        )
        return kwargs

    def create_model(self, args, dataloader):
        """Create TSMixerExt model instance."""
        sample = dataloader["train_loader"].dataset[0]
        num_channels = sample[0].shape[-1]
        extra_channels = sample[1].shape[-1]
        static_channels = sample[3].shape[0]

        return TSMixerExt(
            num_channels=num_channels,
            seq_len=args.seq_len,
            horizon=args.horizon,
            extra_channels=extra_channels,
            hidden_channels=args.hidden_channels,
            static_channels=static_channels,
            activation_fn=args.activation_fn,
            num_blocks=args.num_blocks,
            dropout_rate=args.dropout_rate,
            ff_dim=args.ff_dim,
            normalize_before=args.normalize_before,
            norm_type=args.norm_type,
        )


if __name__ == "__main__":
    run_standard_experiment(TSMixerExtExperiment)

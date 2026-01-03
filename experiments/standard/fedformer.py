from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.fedformer import FEDformer
from src.engines.fedformer_engine import FEDformer_Engine
from src.utils.args import get_fedformer_config

from pathlib import Path
from typing import Dict, Any
import argparse
import torch


class FEDformerStandardExperiment(TorchStandardExperiment):
    """FEDformer-specific training implementation."""

    def get_config_parser(self):
        return get_fedformer_config()

    def get_model_name(self):
        return "fedformer"

    def get_engine_class(self):
        return FEDformer_Engine

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Override to enable time features for FEDformer."""
        kwargs = super().get_dataloader_kwargs(args)
        kwargs["use_time_features"] = getattr(args, "use_time_features", True)
        kwargs["freq"] = getattr(args, "freq", "h")
        kwargs["label_len"] = args.label_len
        return kwargs

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
        """Add label_len to engine kwargs."""
        kwargs = super().get_engine_kwargs(
            args, model, dataloader, scaler, optimizer, scheduler, loss_fn, log_dir, logger
        )
        kwargs["label_len"] = args.label_len
        return kwargs

    def create_model(self, args, dataloader):
        return FEDformer(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            enc_in=args.enc_in,
            dec_in=args.dec_in,
            c_out=args.c_out,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
            embed=args.embed,
            freq=args.freq,
            moving_avg=args.moving_avg,
            # FEDformer-specific
            version=args.version,
            mode_select=args.mode_select,
            modes=args.modes,
            L=args.wavelet_L,
            base=args.wavelet_base,
            cross_activation=args.cross_activation,
            output_attention=args.output_attention,
        )


if __name__ == "__main__":
    run_standard_experiment(FEDformerStandardExperiment)

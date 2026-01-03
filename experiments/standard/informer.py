from typing import Dict, Any
import argparse

from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.informer import Informer
from src.engines.informer_engine import Informer_Engine
from src.utils.args import get_informer_config


"""
WIP, model seems to be training correctly, but fails during evaluation. 
Possible reasons: model is not saved correctly (?), as eval predictions 
resemble those that we get at traingin epoch 0.
Predictions std. seems to collapse during evaluation, but stays normal during
training tasks.
"""


class InformerStandardExperiment(TorchStandardExperiment):
    """Informer-specific training implementation."""

    def get_config_parser(self):
        return get_informer_config()

    def get_model_name(self):
        return "informer"

    def get_engine_class(self):
        return Informer_Engine

    def get_dataloader_kwargs(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Get kwargs for dataloader initialization with time features for Informer."""
        # Get base kwargs from parent class
        kwargs = super().get_dataloader_kwargs(args)

        # Add time feature parameters for Informer
        kwargs.update(
            {
                "use_time_features": getattr(args, "use_time_features", True),
                "freq": getattr(args, "freq", "h"),
                "timeenc": 1 if getattr(args, "embed", "timeF") == "timeF" else 0,
                "embed": getattr(args, "embed", "timeF"),
                "label_len": getattr(args, "label_len", 48),
            }
        )

        return kwargs

    def create_model(self, args, dataloader):
        return Informer(
            # Core sequence parameters
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            # Input/Output dimensions
            enc_in=args.enc_in,
            dec_in=args.dec_in,
            c_out=args.c_out,
            # Model architecture parameters
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            d_ff=args.d_ff,
            # Attention parameters
            factor=args.factor,
            dropout=args.dropout,
            # Distillation (Informer-specific)
            distil=args.distil,
            # Embedding parameters
            embed=args.embed,
            freq=args.freq,
            attn=args.attn,
            # Activation
            activation=args.activation,
            # Output attention
            output_attention=args.output_attention,
            mix=args.mix,
        )


if __name__ == "__main__":
    run_standard_experiment(InformerStandardExperiment)

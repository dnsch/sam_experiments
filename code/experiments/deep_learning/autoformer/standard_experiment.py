# code/experiments/deep_learning/autoformer/standard_training.py

import sys
from pathlib import Path


# Add base training to path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))
import pdb

# pdb.set_trace()

from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)

from src.models.time_series.formers.autoformer import Autoformer
from src.engines.autoformer_engine import Autoformer_Engine
from src.utils.args import get_autoformer_config


class AutoformerStandardExperiment(TorchStandardExperiment):
    """Autoformer-specific training implementation."""

    def get_config_parser(self):
        return get_autoformer_config()

    def get_model_name(self):
        return "autoformer"

    def get_engine_class(self):
        return Autoformer_Engine

    def create_model(self, args, dataloader):
        return Autoformer(
            # Core sequence parameters
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.horizon,
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
            # Decomposition parameter
            moving_avg=args.moving_avg,
            # Attention parameters
            factor=args.factor,
            dropout=args.dropout,
            # Embedding parameters
            embed_type=args.embed_type,
            embed=args.embed,
            freq=args.freq,
            # Activation
            activation=args.activation,
            # Output attention
            output_attention=args.output_attention,
        )


if __name__ == "__main__":
    run_standard_experiment(AutoformerStandardExperiment)

from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.nlinear import NLinear
from src.engines.nlinear_engine import NLinear_Engine
from src.utils.args import get_nlinear_config


class NLinearStandardExperiment(TorchStandardExperiment):
    """NLinear-specific training implementation."""

    def get_config_parser(self):
        return get_nlinear_config()

    def get_model_name(self):
        return "nlinear"

    def get_engine_class(self):
        return NLinear_Engine

    def create_model(self, args, dataloader):
        return NLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
        )


if __name__ == "__main__":
    run_standard_experiment(NLinearStandardExperiment)

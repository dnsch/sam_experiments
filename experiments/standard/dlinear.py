from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.dlinear import DLinear
from src.engines.dlinear_engine import DLinear_Engine
from src.utils.args import get_dlinear_config


class DLinearStandardExperiment(TorchStandardExperiment):
    """DLinear-specific training implementation."""

    def get_config_parser(self):
        return get_dlinear_config()

    def get_model_name(self):
        return "dlinear"

    def get_engine_class(self):
        return DLinear_Engine

    def create_model(self, args, dataloader):
        return DLinear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
            individual=args.individual,
            kernel_size=args.kernel_size,
        )


if __name__ == "__main__":
    run_standard_experiment(DLinearStandardExperiment)

from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.linear import Linear
from src.engines.linear_engine import Linear_Engine
from src.utils.args import get_linear_config


class LinearStandardExperiment(TorchStandardExperiment):
    """Linear-specific training implementation."""

    def get_config_parser(self):
        return get_linear_config()

    def get_model_name(self):
        return "linear"

    def get_engine_class(self):
        return Linear_Engine

    def create_model(self, args, dataloader):
        return Linear(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
        )


if __name__ == "__main__":
    run_standard_experiment(LinearStandardExperiment)

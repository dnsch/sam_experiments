from statsforecast.models import Naive

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)
from src.utils.args import get_naive_config


class NaiveTraining(NixtlaSequentialComparison):
    """Naive-specific training implementation."""

    def get_config_parser(self):
        return get_naive_config()

    def get_model_name(self):
        return "naive"

    def create_statsforecast_model(self, args):
        return Naive(
            alias=args.alias,
        )


if __name__ == "__main__":
    run_sequential_comparison(NaiveTraining)

from statsforecast.models import SeasonalNaive

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)
from src.utils.args import get_seasonal_naive_config


class SeasonalNaiveTraining(NixtlaSequentialComparison):
    """SeasonalNaive-specific training implementation."""

    def get_config_parser(self):
        return get_seasonal_naive_config()

    def get_model_name(self):
        return "seasonal_naive"

    def create_statsforecast_model(self, args):
        return SeasonalNaive(
            season_length=args.season_length,
            alias=args.alias,
        )


if __name__ == "__main__":
    run_sequential_comparison(SeasonalNaiveTraining)

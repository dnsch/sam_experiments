# experiments/sequential_comparison/seasonal_exponential_smoothing/sequential_comparison.py

from statsforecast.models import SeasonalExponentialSmoothingOptimized

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)
from src.utils.args import get_seasonal_exponential_smoothing_config


class SeasonalExponentialSmoothingTraining(NixtlaSequentialComparison):
    """SeasonalExponentialSmoothingOptimized-specific training implementation."""

    def get_config_parser(self):
        return get_seasonal_exponential_smoothing_config()

    def get_model_name(self):
        return "seasonal_exponential_smoothing"

    def create_statsforecast_model(self, args):
        return SeasonalExponentialSmoothingOptimized(
            season_length=args.season_length,
            alias=args.alias,
        )


if __name__ == "__main__":
    run_sequential_comparison(SeasonalExponentialSmoothingTraining)

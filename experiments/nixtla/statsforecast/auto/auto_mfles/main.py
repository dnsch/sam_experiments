from statsforecast.models import AutoMFLES

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)
from src.utils.args import get_automfles_config


class AutoMFLESTraining(NixtlaSequentialComparison):
    """AutoMFLES-specific training implementation."""

    def get_config_parser(self):
        return get_automfles_config()

    def get_model_name(self):
        return "automfles"

    def create_statsforecast_model(self, args):
        return AutoMFLES(
            test_size=args.horizon,
            season_length=args.season_length,
            n_windows=args.n_windows,
            metric=args.metric,
            verbose=args.verbose,
            prediction_intervals=args.prediction_intervals,
        )


if __name__ == "__main__":
    run_sequential_comparison(AutoMFLESTraining)

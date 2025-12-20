from statsforecast.models import HistoricAverage

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)
from src.utils.args import get_historic_average_config


class HistoricAverageTraining(NixtlaSequentialComparison):
    """HistoricAverage-specific training implementation."""

    def get_config_parser(self):
        return get_historic_average_config()

    def get_model_name(self):
        return "historic_average"

    def create_statsforecast_model(self, args):
        return HistoricAverage(
            alias=args.alias,
        )


if __name__ == "__main__":
    run_sequential_comparison(HistoricAverageTraining)

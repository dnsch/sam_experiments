import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[2]))
sys.path.insert(0, str(SCRIPT_DIR.parents[4]))

from src.base.nixtla_sequential_comparison import (
    NixtlaSequentialComparison,
    run_sequential_comparison,
)

from src.utils.args import get_auto_arima_config
from statsforecast.models import AutoARIMA


class AutoARIMATraining(NixtlaSequentialComparison):
    """AutoARIMA-specific training implementation."""

    def get_config_parser(self):
        return get_auto_arima_config()

    def get_model_name(self):
        return "autoarima"

    def create_statsforecast_model(self, args):
        return AutoARIMA(
            season_length=24,
            max_p=3,
            max_q=3,
            max_P=1,
            max_Q=1,
            d=1,
            D=1,
            stepwise=True,
            approximation=True,
            seasonal=True,
            ic="aic",
        )


if __name__ == "__main__":
    run_sequential_comparison(AutoARIMATraining)

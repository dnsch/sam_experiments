import sys
from pathlib import Path
from darts.models import ARIMA as DartsARIMA

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))


class ARIMA:
    """ARIMA model wrapper for consistent interface."""

    def __init__(
        self, horizon=96, p=1, d=1, q=1, P=1, D=1, Q=1, season_length=12, **kwargs
    ):
        self.horizon = horizon

        seasonal_order = (P, D, Q, season_length) if season_length > 1 else None

        self.model = DartsARIMA(p=p, d=d, q=q, seasonal_order=seasonal_order, **kwargs)

    def fit(self, series, **kwargs):
        """Fit the ARIMA model."""
        return self.model.fit(series, **kwargs)

    # TODO: not sure if add horizon here as argument (for more flexibility) or
    # take self.horizon as in other model files
    def predict(self, **kwargs):
        """Make predictions."""
        return self.model.predict(self.horizon, **kwargs)


__all__ = ["ARIMA"]

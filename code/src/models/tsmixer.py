import sys
from pathlib import Path
import torch
# from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))


from src.base.model import BaseModel

"""TSMixer model factory for consistent model interface."""

try:
    from lib.models.tsmixer.torchtsmixer.tsmixer import TSMixer as _TSMixer
except ImportError as e:
    raise ImportError(
        "TSMixer submodule not found. Make sure to initialize submodules with:\n"
        "git submodule update --init --recursive"
    ) from e


class TSMixer(BaseModel):
    """TSMixer adapter that inherits from BaseModel."""

    def __init__(
        self,
        num_channels,
        input_dim,
        output_dim,
        seq_len=12,
        horizon=12,
        activation_fn="relu",
        num_blocks=2,
        dropout_rate=0.1,
        ff_dim=64,
        normalize_before=True,
        norm_type="batch",
        use_revin=False,
    ):
        super().__init__(num_channels, input_dim, output_dim, seq_len, horizon)

        self.tsmixer = _TSMixer(
            sequence_length=seq_len,
            prediction_length=horizon,
            input_channels=input_dim,
            output_channels=output_dim,
            activation_fn=activation_fn,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            ff_dim=ff_dim,
            normalize_before=normalize_before,
            norm_type=norm_type,
            use_revin=use_revin,
        )

    def forward(self, x_hist: torch.Tensor, flatten_output=False) -> torch.Tensor:
        return self.tsmixer(x_hist, flatten_output)


__all__ = ["TSMixer"]

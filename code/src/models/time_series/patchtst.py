import sys
from pathlib import Path
import torch

# from torch import nn
from typing import Optional, Callable
from torch import Tensor
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[3]))


from src.base.model import BaseModel


try:
    from lib.models.PatchTST.PatchTST_supervised.models.PatchTST import (
        Model as _PatchTST,
    )
except ImportError as e:
    raise ImportError(
        "PatchTST submodule not found. Make sure to initialize submodules with:\n"
        "git submodule update --init --recursive"
    ) from e


class PatchTST(BaseModel):
    """
    PatchTST model wrapper
    """

    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,
        **kwargs,
    ):
        # Initialize BaseModel with parameters from configs
        super(PatchTST, self).__init__(
            seq_len=configs.seq_len,
            horizon=configs.pred_len,
        )

        # Store configurations
        self.configs = configs
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.norm = norm
        self.attn_dropout = attn_dropout
        self.act = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.verbose = verbose

        # Initialize the original PatchTST model
        self.model = _PatchTST(
            configs=configs,
            max_seq_len=max_seq_len,
            d_k=d_k,
            d_v=d_v,
            norm=norm,
            attn_dropout=attn_dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            pretrain_head=pretrain_head,
            head_type=head_type,
            verbose=verbose,
            **kwargs,
        )

        # Store model properties for consistency with other models
        self.use_revin = bool(configs.revin)
        self.num_channels = configs.enc_in
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.individual = bool(configs.individual)
        self.decomposition = bool(configs.decomposition)

        # # Initialize weights
        # self._init_weights()

    def forward(self, x, flatten_output=False):
        """
        Forward pass through PatchTST model

        Args:
            x: Input tensor of shape [Batch, Input_length, Channel]
            flatten_output: Whether to flatten the output (for compatibility)

        Returns:
            Predictions of shape [Batch, Output_length, Channel] or flattened
        """
        # PatchTST expects input shape [Batch, Input_length, Channel]
        out = self.model(x)  # Shape: [Batch, Output_length, Channel]

        if flatten_output:
            # Flatten to [Batch, Output_length * Channel] for compatibility
            return out.reshape(out.shape[0], -1)
        else:
            return out

    def __repr__(self):
        """
        String representation of the model
        """
        return (
            f"PatchTST(channels={self.num_channels}, seq_len={self.seq_len}, "
            f"horizon={self.horizon}, patch_len={self.patch_len}, "
            f"layers={self.configs.e_layers}, heads={self.configs.n_heads}, "
            f"d_model={self.configs.d_model}, params={self.param_num():,})"
        )


__all__ = ["PatchTST"]

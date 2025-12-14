import sys
from pathlib import Path
import torch
from typing import Optional
from torch import Tensor
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[3]))

from src.base.model import BaseModel

try:
    from lib.models.PatchTST.PatchTST_supervised.models.PatchTST_test import (
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
        # Core architecture parameters
        enc_in: int,
        seq_len: int,
        pred_len: int,
        e_layers: int = 3,
        n_heads: int = 16,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.0,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        # Patch parameters
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: bool = True,
        # RevIN parameters
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        # Decomposition parameters
        decomposition: bool = False,
        kernel_size: int = 25,
        # Individual parameter
        individual: bool = False,
        # Additional model parameters
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
        # Initialize BaseModel
        super(PatchTST, self).__init__(
            seq_len=seq_len,
            horizon=pred_len,
        )

        # Store all parameters
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.e_layers = e_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.individual = individual

        # Additional parameters
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

        # Initialize the original PatchTST model directly with parameters
        self.model = _PatchTST(
            enc_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            decomposition=decomposition,
            kernel_size=kernel_size,
            individual=individual,
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

        # Store model properties for consistency
        self.use_revin = bool(revin)
        self.num_channels = enc_in

    def forward(self, x, flatten_output=False):
        """
        Forward pass through PatchTST model

        Args:
            x: Input tensor of shape [Batch, Input_length, Channel]
            flatten_output: Whether to flatten the output (for compatibility)

        Returns:
            Predictions of shape [Batch, Output_length, Channel] or flattened
        """
        out = self.model(x)  # Shape: [Batch, Output_length, Channel]

        if flatten_output:
            return out.reshape(out.shape[0], -1)
        else:
            return out

    def __repr__(self):
        """String representation of the model"""
        return (
            f"PatchTST(channels={self.num_channels}, seq_len={self.seq_len}, "
            f"horizon={self.pred_len}, patch_len={self.patch_len}, "
            f"layers={self.e_layers}, heads={self.n_heads}, "
            f"d_model={self.d_model}, params={self.param_num():,})"
        )


__all__ = ["PatchTST"]

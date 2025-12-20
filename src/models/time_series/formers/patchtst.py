__all__ = ["PatchTST"]

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[0]))
import pdb

# pdb.set_trace()
from src.base.model import BaseModel
from formers.layers.PatchTST_backbone import PatchTST_backbone
from formers.layers.PatchTST_layers import series_decomp


class PatchTST(BaseModel):
    def __init__(
        self,
        # Core architecture parameters
        enc_in: int,
        seq_len: int,
        pred_len: int,
        e_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        fc_dropout: float,
        head_dropout: float,
        # Patch parameters
        patch_len: int,
        stride: int,
        padding_patch: bool,
        # RevIN parameters
        revin: bool,
        affine: bool,
        subtract_last: bool,
        # Decomposition parameters
        decomposition: bool,
        kernel_size: int,
        # Individual parameter
        individual: bool,
        # Additional parameters
        max_seq_len: Optional[int] = 1024,  # legacy, unused, but kept for compatibility
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = "BatchNorm",  # legacy, unused, but kept for compatibility, TODO: extend that param and allow for layer norm
        attn_dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",  # legacy, unused, but kept for compatibility
        padding_var: Optional[int] = None,  # legacy, unused, but kept for compatibility
        attn_mask: Optional[
            Tensor
        ] = None,  # legacy, unused, but kept for compatibility
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        verbose: bool = False,  # legacy, unused, but kept for compatibility
        **kwargs,
    ):
        super().__init__()

        # Store parameters
        c_in = enc_in
        context_window = seq_len
        target_window = pred_len

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=e_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs,
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=e_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs,
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=e_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                act=act,
                key_padding_mask=key_padding_mask,
                padding_var=padding_var,
                attn_mask=attn_mask,
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
                **kwargs,
            )

    # def forward(self, x):  # x: [Batch, Input length, Channel]
    def forward(self, x, flatten_output=False):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = (
                res_init.permute(0, 2, 1),
                trend_init.permute(0, 2, 1),
            )  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        # TODO: maybe get rid of these in every model
        if flatten_output:
            # Flatten to [Batch, Output_length * Channel] for compatibility
            return x.reshape(x.shape[0], -1)
        else:
            return x

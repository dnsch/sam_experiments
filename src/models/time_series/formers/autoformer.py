import torch
import torch.nn as nn

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[0]))

from src.base.model import BaseModel
from formers.layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from formers.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from formers.layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
)


class Autoformer(BaseModel):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(
        self,
        # Core sequence parameters
        seq_len: int,
        label_len: int,
        pred_len: int,
        # Input/Output dimensions
        enc_in: int,
        dec_in: int,
        c_out: int,
        # Model architecture parameters
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        # Decomposition parameter
        moving_avg: int,
        # Attention parameters
        factor: int,
        dropout: float,
        # Embedding parameters
        embed_type: int = 0,  # 0: wo_pos, 1: with_pos, 2: wo_pos, 3: wo_temp, 4: wo_pos_temp
        embed: str = "timeF",
        freq: str = "h",
        # Activation
        activation: str = "gelu",
        # Output attention
        output_attention: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Store parameters
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(
                enc_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )
        elif embed_type == 1:
            self.enc_embedding = DataEmbedding(
                enc_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.dec_embedding = DataEmbedding(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )
        elif embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(
                enc_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )
        elif embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(
                enc_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.dec_embedding = DataEmbedding_wo_temp(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )
        elif embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(
                enc_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.dec_embedding = DataEmbedding_wo_pos_temp(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        enc_self_mask=None,
        dec_self_mask=None,
        dec_enc_mask=None,
        flatten_output=False,
    ):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len :, :], zeros], dim=1
        )

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        # final
        dec_out = trend_part + seasonal_part
        output = dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if flatten_output:
            # Flatten to [Batch, Output_length * Channel] for compatibility
            output = output.reshape(output.shape[0], -1)

        if self.output_attention:
            return output, attns
        else:
            return output

import torch.nn as nn

from src.base.model import BaseModel
from src.models.time_series.formers.layers.Embed import (
    DataEmbedding,
    DataEmbedding_wo_pos,
    DataEmbedding_wo_pos_temp,
    DataEmbedding_wo_temp,
)
from src.models.time_series.formers.layers.SelfAttention_Family import (
    ProbAttention,
    AttentionLayer,
)
from src.models.time_series.formers.layers.Transformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    ConvLayer,
)


class Informer(BaseModel):
    """
    Informer with ProbSparse attention in O(LlogL) complexity.
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
        # Attention parameters
        factor: int,
        dropout: float,
        # Distillation (Informer-specific)
        distil: bool = True,
        # Embedding parameters
        embed_type: int = 0,
        embed: str = "timeF",
        freq: str = "h",
        # Activation
        activation: str = "gelu",
        # Output attention
        output_attention: bool = False,
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        self.label_len = label_len
        self.output_attention = output_attention
        self.distil = distil

        # Embedding
        if embed_type == 0:
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
                    AttentionLayer(
                        ProbAttention(
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
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
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
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
        )

        output = dec_out[:, -self.pred_len :, :]  # [B, L, D]

        if flatten_output:
            output = output.reshape(output.shape[0], -1)

        if self.output_attention:
            return output, attns
        else:
            return output

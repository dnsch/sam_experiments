import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import BaseModel
from src.models.time_series.formers.layers.Embed import DataEmbedding_wo_pos
from src.models.time_series.formers.layers.FourierCorrelation import (
    FourierBlock,
    FourierCrossAttention,
)
from src.models.time_series.formers.layers.MultiWaveletCorrelation import (
    MultiWaveletCross,
    MultiWaveletTransform,
)
from src.models.time_series.formers.layers.AutoCorrelation import AutoCorrelationLayer
from src.models.time_series.formers.layers.Autoformer_EncDec import (
    Encoder,
    Decoder,
    EncoderLayer,
    DecoderLayer,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)


class FEDformer(BaseModel):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity.
    Paper: https://arxiv.org/abs/2201.12740
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        label_len: int,
        enc_in: int,
        dec_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        dropout: float = 0.05,
        activation: str = "gelu",
        embed: str = "timeF",
        freq: str = "h",
        moving_avg: int = 25,
        # FEDformer-specific
        version: str = "Fourier",
        mode_select: str = "random",
        modes: int = 32,
        # Wavelet-specific
        L: int = 1,
        base: str = "legendre",
        cross_activation: str = "tanh",
        output_attention: bool = False,
        **kwargs,
    ):
        super().__init__(seq_len=seq_len, pred_len=pred_len)

        self.label_len = label_len
        self.output_attention = output_attention
        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomposition
        if isinstance(moving_avg, list):
            self.decomp = series_decomp_multi(moving_avg)
        else:
            self.decomp = series_decomp(moving_avg)

        # Embedding (without positional encoding)
        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        # Build attention layers based on version
        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=L, base=base)
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len,
                modes=modes,
                ich=d_model,
                base=base,
                activation=cross_activation,
            )
        else:  # Fourier
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=seq_len,
                modes=modes,
                mode_select_method=mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                seq_len=seq_len // 2 + pred_len,
                modes=modes,
                mode_select_method=mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=seq_len // 2 + pred_len,
                seq_len_kv=seq_len,
                modes=modes,
                mode_select_method=mode_select,
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
                    d_model,
                    c_out,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
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
    ):
        # Get device from input
        device = x_enc.device

        # Decomposition init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)

        # Decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len))

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            trend=trend_init,
        )

        # Final output
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        else:
            return dec_out[:, -self.pred_len :, :]  # [B, L, D]

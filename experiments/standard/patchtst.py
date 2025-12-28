from src.base.torch_standard_experiment import (
    TorchStandardExperiment,
    run_standard_experiment,
)
from src.models.time_series.formers.patchtst import PatchTST
from src.engines.patchtst_engine import PatchTST_Engine
from src.utils.args import get_patchtst_config


class PatchTSTExperiment(TorchStandardExperiment):
    """PatchTST-specific training implementation."""

    def get_config_parser(self):
        return get_patchtst_config()

    def get_model_name(self):
        return "patchtst"

    def get_engine_class(self):
        return PatchTST_Engine

    def create_model(self, args, dataloader):
        return PatchTST(
            # Core architecture parameters
            enc_in=args.enc_in,
            seq_len=args.seq_len,
            pred_len=args.horizon,
            e_layers=args.e_layers,
            n_heads=args.n_heads,
            d_model=args.d_model,
            d_ff=args.d_ff,
            dropout=args.dropout,
            fc_dropout=args.fc_dropout,
            head_dropout=args.head_dropout,
            # Patch parameters
            patch_len=args.patch_len,
            stride=args.stride,
            padding_patch=args.padding_patch,
            # Decomposition parameters
            decomposition=args.decomposition,
            kernel_size=args.kernel_size,
            # Individual parameter
            individual=args.individual,
            # Additional parameters
            max_seq_len=args.seq_len,
            d_k=args.d_k,
            d_v=args.d_v,
            norm=args.norm,
            attn_dropout=args.attn_dropout,
            act=args.activation,
            key_padding_mask=args.key_padding_mask,
            padding_var=args.padding_var,
            attn_mask=args.attn_mask,
            res_attention=args.res_attention,
            pre_norm=args.pre_norm,
            store_attn=args.store_attn,
            pe=args.pe,
            learn_pe=args.learn_pe,
            pretrain_head=args.pretrain_head,
            head_type=args.head_type,
            verbose=args.verbose,
        )

    def get_revin_num_features(self, args):
        """PatchTST uses enc_in for number of channels."""
        return args.enc_in


if __name__ == "__main__":
    run_standard_experiment(PatchTSTExperiment)

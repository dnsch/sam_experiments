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

"""TSMixer model factory for consistent model interface."""

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
    PatchTST model wrapper that inherits from BaseModel for consistency
    with the project structure.
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
            node_num=None,  # PatchTST doesn't use node_num
            input_dim=configs.enc_in,
            output_dim=configs.enc_in,  # Typically same as input for time series
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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Xavier/Glorot Uniform initialization
        similar to SAMFormer for consistency
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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

    def get_model_info(self):
        """
        Get comprehensive information about the model
        """
        info = {
            "model_type": "PatchTST",
            "total_parameters": self.param_num(),
            "input_channels": self.num_channels,
            "sequence_length": self.seq_len,
            "prediction_horizon": self.horizon,
            "patch_length": self.patch_len,
            "stride": self.stride,
            "use_revin": self.use_revin,
            "individual_heads": self.individual,
            "decomposition": self.decomposition,
            "transformer_layers": self.configs.e_layers,
            "attention_heads": self.configs.n_heads,
            "model_dimension": self.configs.d_model,
            "feedforward_dimension": self.configs.d_ff,
            "dropout_rate": self.configs.dropout,
            "fc_dropout_rate": self.configs.fc_dropout,
            "head_dropout_rate": self.configs.head_dropout,
        }
        return info

    def get_attention_weights(self):
        """
        Extract attention weights if available
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            # Try to extract attention weights from the transformer encoder
            attention_weights = []
            for layer in self.model.model.encoder.layers:
                if hasattr(layer, "self_attn") and hasattr(
                    layer.self_attn, "attention_weights"
                ):
                    attention_weights.append(layer.self_attn.attention_weights)
            return attention_weights
        return None

    def set_store_attention(self, store_attn=True):
        """
        Enable/disable attention weight storage
        """
        self.store_attn = store_attn
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            for layer in self.model.model.encoder.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.store_attn = store_attn

    def reset_parameters(self):
        """
        Reset all model parameters
        """
        self._init_weights()

    def get_patch_info(self):
        """
        Get information about patch configuration
        """
        # Calculate number of patches
        seq_len = self.seq_len
        patch_len = self.patch_len
        stride = self.stride

        if self.configs.padding_patch == "end":
            # Calculate patches with end padding
            n_patches = (seq_len - patch_len) // stride + 1
            if (seq_len - patch_len) % stride != 0:
                n_patches += 1
        else:
            # Calculate patches without padding
            n_patches = (seq_len - patch_len) // stride + 1

        patch_info = {
            "patch_length": patch_len,
            "stride": stride,
            "num_patches": n_patches,
            "padding_patch": self.configs.padding_patch,
            "input_sequence_length": seq_len,
            "effective_patch_coverage": min(
                seq_len, (n_patches - 1) * stride + patch_len
            ),
        }

        return patch_info

    def freeze_backbone(self):
        """
        Freeze the backbone (encoder) parameters
        """
        if hasattr(self.model, "model"):
            for param in self.model.model.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreeze the backbone (encoder) parameters
        """
        if hasattr(self.model, "model"):
            for param in self.model.model.parameters():
                param.requires_grad = True

    def freeze_head(self):
        """
        Freeze the prediction head parameters
        """
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = False

    def unfreeze_head(self):
        """
        Unfreeze the prediction head parameters
        """
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True

    def get_layer_wise_lr_groups(self, base_lr=1e-3, decay_factor=0.9):
        """
        Get parameter groups for layer-wise learning rate decay
        """
        param_groups = []

        # Head parameters (highest learning rate)
        if hasattr(self.model, "head"):
            param_groups.append(
                {"params": self.model.head.parameters(), "lr": base_lr, "name": "head"}
            )

        # Encoder layers (decreasing learning rate)
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            layers = self.model.model.encoder.layers
            for i, layer in enumerate(reversed(layers)):
                lr = base_lr * (decay_factor ** (i + 1))
                param_groups.append(
                    {
                        "params": layer.parameters(),
                        "lr": lr,
                        "name": f"encoder_layer_{len(layers) - 1 - i}",
                    }
                )

        # Embedding parameters (lowest learning rate)
        if hasattr(self.model, "model"):
            remaining_params = []
            for name, param in self.model.model.named_parameters():
                if "encoder.layers" not in name and "head" not in name:
                    remaining_params.append(param)

            if remaining_params:
                param_groups.append(
                    {
                        "params": remaining_params,
                        "lr": base_lr * (decay_factor ** len(layers)),
                        "name": "embeddings",
                    }
                )

        return param_groups

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

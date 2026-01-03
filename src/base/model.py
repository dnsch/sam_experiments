import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=96):
        super(BaseModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

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

    # TODO: get_experiment_summary might be more fitting
    # might also make sense to put this somewhere else
    def get_model_summary(self, args):
        """
        Generate a formatted summary using cli args

        Args:
            args: argparse.Namespace object containing all arguments
        """

        model_name = args.model_name

        # Terminal width chars
        width = 80
        top_border_with = width - 2

        lines = []

        # Top border
        lines.append(" " + "=" * top_border_with)

        # Title
        title = f"MODEL CONFIGURATION: {model_name.upper()}"
        lines.append(title.center(top_border_with))
        lines.append(" " + "=" * top_border_with)

        # Parameter count
        param_info = f"Total Parameters: {self.param_num():,}"
        lines.append(param_info.center(width))
        lines.append(" " + "=" * top_border_with)

        # Column widths (adjust to fit within 80 chars with borders)
        # Format: | Argument (35) | Value (40) |
        arg_col_width = 36
        val_col_width = 37

        # Table header
        lines.append(f"| {'Argument'.ljust(arg_col_width)} | {'Value'.ljust(val_col_width)} |")
        lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")

        args_dict = vars(args)

        parser = args._parser

        if parser is not None:
            # Automatically extract groups from parser
            first_group = True
            for group in parser._action_groups:
                # Skip default argparse groups we don't want to display
                if group.title in [
                    "positional arguments",
                    "optional arguments",
                    "options",
                ]:
                    continue

                # Get all arguments in this group
                group_args = []
                for action in group._group_actions:
                    arg_name = action.dest
                    if arg_name in args_dict and arg_name != "help":
                        value = args_dict[arg_name]
                        group_args.append((arg_name, value))

                # Only add group if it has arguments
                if group_args:
                    # Add separator line between groups (except before first group)
                    if not first_group:
                        lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")
                    first_group = False

                    # Group header
                    group_header = f"*** {group.title} ***"
                    lines.append(f"| {group_header.ljust(arg_col_width + val_col_width + 3)} |")
                    lines.append(f"|{'-' * (arg_col_width + 3)}{'-' * (val_col_width + 2)}|")

                    # Group arguments
                    for name, value in group_args:
                        # Format value
                        if isinstance(value, bool):
                            value_str = "True" if value else "False"
                        elif value is None:
                            value_str = "None"
                        elif isinstance(value, (list, tuple)):
                            value_str = ", ".join(map(str, value))
                        else:
                            value_str = str(value)

                        # Truncate if too long and add ellipsis
                        if len(value_str) > val_col_width:
                            value_str = value_str[: val_col_width - 3] + "..."

                        lines.append(
                            f"| {name.ljust(arg_col_width)} | {value_str.ljust(val_col_width)} |"
                        )
        else:
            # Fallback: show all arguments in a single group
            for name, value in sorted(args_dict.items()):
                # Format value
                if isinstance(value, bool):
                    value_str = "✓" if value else "✗"
                elif value is None:
                    value_str = "None"
                elif isinstance(value, (list, tuple)):
                    value_str = ", ".join(map(str, value))
                else:
                    value_str = str(value)

                # Truncate if too long and add ellipsis
                if len(value_str) > val_col_width:
                    value_str = value_str[: val_col_width - 3] + "..."

                lines.append(f"| {name.ljust(arg_col_width)} | {value_str.ljust(val_col_width)} |")

        # Bottom border
        lines.append("=" * width)

        return "\n".join(lines)

    def print_model_summary(self, args, logger=None):
        """
        Print the model summary to console or logger.

        Args:
            args: argparse.Namespace object containing all arguments
            logger: Optional logger object. If None, prints to console
        """
        summary = self.get_model_summary(args)

        if logger:
            logger.info("\n" + summary)
        else:
            print(summary)

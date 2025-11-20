import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# TODO: change paths here, don't need all of em

sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))


from src.models.time_series.patchtst import PatchTST
from src.engines.patchtst_engine import PatchTST_Engine
from src.utils.args import get_patchtst_config
from src.utils.dataloader import (
    SamformerDataloader,
)
from src.utils.logging import get_logger

# TODO: write this in a separate class
from src.utils.samformer_utils.sam import SAM

from src.utils.functions import (
    set_seed,
    load_optimizer,
    compute_top_eigenvalue_and_eigenvector,
    compute_dominant_hessian_directions,
    save_eigenvectors_to_hdf5,
)

import torch
import argparse

torch.set_num_threads(3)
from lib.utils.pyhessian.density_plot import get_esd_plot
from lib.optimizers.gsam.gsam.gsam import GSAM
from lib.optimizers.gsam.gsam.scheduler import LinearScheduler


class PatchTSTConfig:
    """Configuration class to mimic the args structure expected by PatchTST"""

    def __init__(self, args):
        # Map args to the expected PatchTST configuration
        self.enc_in = args.num_channels
        self.seq_len = args.seq_len
        self.pred_len = args.horizon
        self.e_layers = args.e_layers
        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        self.fc_dropout = args.fc_dropout
        self.head_dropout = args.head_dropout
        self.individual = args.individual
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.padding_patch = args.padding_patch
        self.revin = args.revin
        self.affine = args.affine
        self.subtract_last = args.subtract_last
        self.decomposition = args.decomposition
        self.kernel_size = args.kernel_size


def get_config():
    parser = get_patchtst_config()
    args = parser.parse_args()

    args.model_name = "patchtst"

    base_dir = SCRIPT_DIR.parents[2] / "results"

    # Logger
    if args.no_sam:
        log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}/".format(
            base_dir,
            args.model_name + "_without_sam",
            args.dataset,
            args.seq_len,
            args.horizon,
            args.batch_size,
        )
    else:
        log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}_rho_{}/".format(
            base_dir,
            args.model_name,
            args.dataset,
            args.seq_len,
            args.horizon,
            args.batch_size,
            args.rho,
        )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def run_experiments_on_dataloader_list(
    dataloader_instance,
    dataloader_list,
    args,
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    log_dir,
    logger,
):
    """
    Execute training/evaluation for each dataloader in dataloader_list.
    """
    # Get the scaler list
    scaler_list = dataloader_instance.get_scaler_list()

    results = []

    # Iterate through each dataloader
    for idx, dataloader in enumerate(dataloader_list):
        print(f"\n{'=' * 60}")
        print(f"Processing Experiment {idx + 1}/{len(dataloader_list)}")
        print(f"{'=' * 60}\n")

        # Get the corresponding scaler for this dataloader
        scaler = scaler_list[idx] if idx < len(scaler_list) else None

        # Create experiment-specific log directory
        experiment_log_dir = log_dir / f"experiment_{idx}"

        # Create the engine
        engine = PatchTST_Engine(
            device=args.device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            lrate=args.lrate,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            clip_grad_value=args.clip_grad_value,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_dir=experiment_log_dir,
            logger=logger,
            seed=args.seed,
            batch_size=args.batch_size,
            num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
            pred_len=args.horizon,
            no_sam=args.no_sam,
            use_revin=args.revin,
            gsam=args.gsam,
        )

        # Run train or test based on mode
        if args.mode == "train":
            result = engine.train()
        elif args.mode == "test":
            result = engine.evaluate(args.mode)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        results.append(result)

        print(f"\nCompleted Experiment {idx + 1}/{len(dataloader_list)}\n")

    return results


def main():
    args, log_dir, logger = get_config()

    set_seed(args.seed)

    dataset_name = args.dataset
    time_increment = 1
    sequential_comparison = False

    dataloader_instance = SamformerDataloader(
        dataset_name,
        args,
        logger,
        time_increment,
        sequential_comparison=sequential_comparison,
    )

    if sequential_comparison:
        dataloader_list = dataloader_instance.get_dataloader()

        # Create PatchTST configuration
        patchtst_config = PatchTSTConfig(args)

        model = PatchTSTModel(
            configs=patchtst_config,
            max_seq_len=args.max_seq_len,
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
    else:
        dataloader = dataloader_instance.get_dataloader()

        # Create PatchTST configuration
        patchtst_config = PatchTSTConfig(args)

        model = PatchTST(
            patchtst_config,
            max_seq_len=args.max_seq_len,
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

    optimizer = load_optimizer(model, args, logger)

    # Setup learning rate scheduler and SAM if needed
    lr_scheduler = None
    if not args.no_sam:
        if args.gsam:
            lr_scheduler = LinearScheduler(
                T_max=args.max_epochs * len(dataloader["train_loader"]),
                max_value=args.lrate,
                min_value=args.lrate * 0.01,
                optimizer=optimizer,
            )
            rho_scheduler = LinearScheduler(
                T_max=args.max_epochs * len(dataloader["train_loader"]),
                max_value=args.gsam_rho_max,
                min_value=args.gsam_rho_min,
            )
            optimizer = GSAM(
                params=model.parameters(),
                base_optimizer=optimizer,
                model=model,
                gsam_alpha=args.gsam_alpha,
                rho_scheduler=rho_scheduler,
                adaptive=args.gsam_adaptive,
            )
        else:
            base_optimizer_class = getattr(torch.optim, args.optimizer)
            optimizer = SAM(
                params=model.parameters(),
                base_optimizer=base_optimizer_class,
                rho=args.rho,
                lr=args.lrate,
                weight_decay=args.wdecay,
            )
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=5,
                T_mult=1,
                eta_min=1e-6,
                last_epoch=-1,
            )

    loss_fn = torch.nn.MSELoss()

    if sequential_comparison:
        results = run_experiments_on_dataloader_list(
            dataloader_instance=dataloader_instance,
            dataloader_list=dataloader_list,
            args=args,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_dir=log_dir,
            logger=logger,
        )

        for idx, result in enumerate(results):
            print(f"Experiment {idx}: {result}")

    else:
        scaler = dataloader_instance.get_scaler()

        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )

        engine = PatchTST_Engine(
            device=args.device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            lrate=args.lrate,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            clip_grad_value=args.clip_grad_value,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_dir=log_dir,
            logger=logger,
            seed=args.seed,
            batch_size=args.batch_size,
            num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
            pred_len=args.horizon,
            no_sam=args.no_sam,
            use_revin=args.revin,
            gsam=args.gsam,
        )

        if args.mode == "train":
            result = engine.train()
            print(f"Result: {result}")
        elif args.mode == "test":
            result = engine.evaluate(args.mode)
            print(f"Result: {result}")

        # Hessian analysis (optional)
        if hasattr(args, "hessian_analysis") and args.hessian_analysis:
            top_eigenvalue, top_eigenvector = compute_top_eigenvalue_and_eigenvector(
                model, loss_fn, dataloader["train_loader"]
            )
            print(f"Max Eigenvalue: {top_eigenvalue}")

            from lib.utils.pyhessian.pyhessian import hessian

            hessian_comp = hessian(
                model, loss_fn, dataloader=dataloader["train_loader"], cuda=args.device
            )
            density_eigen, density_weight = hessian_comp.density()
            get_esd_plot(density_eigen, density_weight)

            if args.hessian_directions:
                max_ev, max_evec, min_ev, min_evec = (
                    compute_dominant_hessian_directions(
                        model,
                        loss_fn,
                        dataloader["train_loader"],
                        tol=1e-4,
                    )
                )
                save_eigenvectors_to_hdf5(
                    args=args,
                    net=model,
                    max_evec=max_evec,
                    min_evec=min_evec,
                    output_dir=log_dir / "hessian_directions",
                )


if __name__ == "__main__":
    main()

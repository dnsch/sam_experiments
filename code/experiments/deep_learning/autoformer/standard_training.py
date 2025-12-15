import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))

from src.models.time_series.formers.autoformer import Autoformer
from src.engines.autoformer_engine import Autoformer_Engine
from src.utils.args import get_autoformer_config
from src.utils.dataloader import SamformerDataloader
from src.utils.logging import get_logger
from src.utils.samformer_utils.sam import SAM
from src.utils.reproducibility import set_seed
from src.utils.model_utils import load_optimizer

import torch

torch.set_num_threads(3)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lib.utils.pyhessian.density_plot import get_esd_plot
from lib.optimizers.gsam.gsam.gsam import GSAM
from lib.optimizers.gsam.gsam.scheduler import LinearScheduler


def get_config():
    parser = get_autoformer_config()
    args = parser.parse_args()
    args._parser = parser

    args.model_name = "autoformer"

    base_dir = SCRIPT_DIR.parents[3] / "results"

    # Logger
    if args.sam:
        log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}_rho_{}/".format(
            base_dir,
            args.model_name + "SAM",
            args.dataset,
            args.seq_len,
            args.horizon,
            args.batch_size,
            args.rho,
        )
    elif args.gsam:
        log_dir = (
            "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}_gsam_alpha_{}_rho_max_{}/".format(
                base_dir,
                args.model_name + "GSAM",
                args.dataset,
                args.seq_len,
                args.horizon,
                args.batch_size,
                args.gsam_alpha,
                args.gsam_rho_max,
            )
        )
    else:
        log_dir = "{}/{}/{}/seq_len_{}_pred_len_{}_bs_{}/".format(
            base_dir,
            args.model_name,
            args.dataset,
            args.seq_len,
            args.horizon,
            args.batch_size,
        )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_dir, __name__, "record_s{}.log".format(args.seed))
    logger.info(args)

    return args, log_dir, logger


def main():
    args, log_dir, logger = get_config()

    set_seed(args.seed)

    dataset_name = args.dataset
    time_increment = 1
    sequential_comparison = False

    # Initialize dataloader
    dataloader_instance = SamformerDataloader(
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.horizon,
        seed=args.seed,
        time_increment=time_increment,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        sequential_comparison=sequential_comparison,
    )

    dataloader = dataloader_instance.get_dataloader()

    # Create Autoformer model
    model = Autoformer(
        # Core sequence parameters
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.horizon,
        # Input/Output dimensions
        enc_in=args.enc_in,
        dec_in=args.dec_in,
        c_out=args.c_out,
        # Model architecture parameters
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        # Decomposition parameter
        moving_avg=args.moving_avg,
        # Attention parameters
        factor=args.factor,
        dropout=args.dropout,
        # Embedding parameters
        embed_type=args.embed_type,
        embed=args.embed,
        freq=args.freq,
        # Activation
        activation=args.activation,
        # Output attention
        output_attention=args.output_attention,
    )

    # Load optimizer
    optimizer = load_optimizer(model, args, logger)
    model.print_model_summary(args, logger)

    # Setup learning rate scheduler and SAM/GSAM if needed
    lr_scheduler = None
    if args.sam:
        base_optimizer_class = getattr(torch.optim, args.optimizer)
        optimizer = SAM(
            params=model.parameters(),
            base_optimizer=base_optimizer_class,
            rho=args.rho,
            lr=args.lrate,
            weight_decay=args.wdecay,
        )
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )

    elif args.gsam:
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
        # Standard learning rate scheduler
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1,
        )

    # Loss function
    loss_fn = torch.nn.MSELoss()

    # Get scaler
    scaler = dataloader_instance.get_scaler()

    # Create engine
    engine = Autoformer_Engine(
        device=args.device,
        model=model,
        dataloader=dataloader,
        scaler=scaler,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        sam=args.sam,
        gsam=args.gsam,
        scheduler=lr_scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        metrics=["mse"],  # Track these metrics
    )

    # Train or test
    if args.mode == "train":
        result = engine.train()
        logger.info(f"Training completed. Result: {result}")
    elif args.mode == "test":
        result = engine.evaluate(args.mode)
        logger.info(f"Testing completed. Result: {result}")

    # Hessian analysis (optional)
    if hasattr(args, "hessian_analysis") and args.hessian_analysis:
        from lib.utils.pyhessian.pyhessian import hessian

        logger.info("Computing Hessian analysis...")

        hessian_comp = hessian(
            model, loss_fn, dataloader=dataloader["train_loader"], cuda=args.device
        )
        density_eigen, density_weight = hessian_comp.density()
        get_esd_plot(density_eigen, density_weight)

        if args.hessian_directions:
            from src.utils.hessian_utils import (
                compute_dominant_hessian_directions,
                save_eigenvectors_to_hdf5,
            )

            max_ev, max_evec, min_ev, min_evec = compute_dominant_hessian_directions(
                model,
                loss_fn,
                dataloader["train_loader"],
                tol=1e-4,
            )
            save_eigenvectors_to_hdf5(
                args=args,
                net=model,
                max_evec=max_evec,
                min_evec=min_evec,
                output_dir=log_dir / "hessian_directions",
            )
            logger.info("Hessian directions saved.")


if __name__ == "__main__":
    main()

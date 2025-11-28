import sys
from pathlib import Path
import time

from matplotlib.pyplot import plot

SCRIPT_DIR = Path(__file__).resolve().parent
# TODO: change paths here, don't need all of em
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))

from src.models.time_series.samformer import SAMFormer

from src.engines.samformer_engine import SAMFormer_Engine
from src.utils.args import get_samformer_config
from src.utils.dataloader import (
    SamformerDataloader,
)
from src.utils.logging import get_logger
from src.utils.samformer_utils.sam import SAM

from src.utils.reproducibility import set_seed
from src.utils.experiment_utils import run_experiments_on_dataloader_list
from src.utils.model_utils import load_optimizer

# from src.utils.functions import (
#     compute_top_eigenvalue_and_eigenvector,
#     compute_dominant_hessian_directions,
#     save_eigenvectors_to_hdf5,
# )

import torch


torch.set_num_threads(3)
from lib.utils.pyhessian.density_plot import get_esd_plot
from lib.optimizers.gsam.gsam.gsam import GSAM
from lib.optimizers.gsam.gsam.scheduler import LinearScheduler


def get_config():
    parser = get_samformer_config()
    args = parser.parse_args()
    args._parser = parser

    args.model_name = "samformer"

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


def main():
    args, log_dir, logger = get_config()

    set_seed(args.seed)

    # TODO: add this as parameter
    time_increment = 1
    # TODO: add this as parameter
    sequential_comparison = False
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

    # TODO:add as parameter
    args.plot_attention = True
    # TODO: add this as parameter
    if sequential_comparison:
        dataloader_list = dataloader_instance.get_dataloader()

        model = SAMFormer(
            num_channels=dataloader_list[0]["train_loader"].dataset[0][0].shape[0],
            seq_len=args.seq_len,
            hid_dim=args.hid_dim,
            horizon=args.horizon,
            use_revin=args.use_revin,
            plot_attention=args.plot_attention,
        )
    else:
        dataloader = dataloader_instance.get_dataloader()
        model = SAMFormer(
            num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
            seq_len=args.seq_len,
            hid_dim=args.hid_dim,
            horizon=args.horizon,
            use_revin=args.use_revin,
            plot_attention=args.plot_attention,
        )

    optimizer = load_optimizer(model, args, logger)
    model.print_model_summary(args, logger)
    # TODO: add option to choose lr_scheduler
    # also lr_scheduler=None option
    from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

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
            # base_optimizer = base_optimizer_class(
            #     model.parameters(), lr=args.lrate, weight_decay=args.wdecay
            # )

            # lr_scheduler = CosineAnnealingWarmRestarts(
            #     optimizer=base_optimizer,  # Use base optimizer
            #     T_0=5,
            #     T_mult=1,
            #     eta_min=1e-5,
            #     last_epoch=-1,
            # )
            optimizer = SAM(
                params=model.parameters(),
                base_optimizer=base_optimizer_class,
                rho=args.rho,
                lr=args.lrate,
                weight_decay=args.wdecay,
            )
            lr_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=5,  # Restart every 5 epochs (like max_epochs=5 in your example)
                T_mult=1,  # Keep the same cycle length
                eta_min=1e-6,  # Minimum learning rate
                last_epoch=-1,
            )
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer.base_optimizer, T_max=args.max_epochs, eta_min=1e-6
            # )

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

        ### Optionally, aggregate or analyze results

        for idx, result in enumerate(results):
            print(f"Experiment {idx}: {result}")

    else:
        scaler = dataloader_instance.get_scaler()
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=1,  # Keep the same cycle length
            eta_min=1e-6,  # Minimum learning rate
            last_epoch=-1,
        )
        # engine = SAMFormer_Engine(
        #     device=args.device,
        #     model=model,
        #     dataloader=dataloader,
        #     scaler=scaler,
        #     # scaler=None,
        #     loss_fn=loss_fn,
        #     lrate=args.lrate,
        #     optimizer=optimizer,
        #     scheduler=lr_scheduler,
        #     clip_grad_value=args.clip_grad_value,
        #     # clip_grad_value=4,
        #     max_epochs=args.max_epochs,
        #     patience=args.patience,
        #     log_dir=log_dir,
        #     logger=logger,
        #     seed=args.seed,
        #     batch_size=args.batch_size,
        #     num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
        #     pred_len=args.horizon,
        #     no_sam=args.no_sam,
        #     use_revin=args.use_revin,
        #     gsam=args.gsam,
        #     plot_attention=args.plot_attention,
        # )
        engine = SAMFormer_Engine(
            device=args.device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            primary_metric="mse",  # Use MSE for early stopping
            lrate=args.lrate,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            clip_grad_value=args.clip_grad_value,
            # clip_grad_value=4,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_dir=log_dir,
            logger=logger,
            seed=args.seed,
            batch_size=args.batch_size,
            num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
            pred_len=args.horizon,
            no_sam=args.no_sam,
            use_revin=args.use_revin,
            gsam=args.gsam,
            plot_attention=args.plot_attention,
            metrics=["mse", "mape", "rmse"],  # Track these metrics
        )

        if args.mode == "train":
            result = engine.train()
            print(f"Result: {result}")
        elif args.mode == "test":
            result = engine.evaluate(args.mode)
            print(f"Result: {result}")

        top_eigenvalue, top_eigenvector = compute_top_eigenvalue_and_eigenvector(
            model, loss_fn, dataloader["train_loader"]
        )
        print(f"Max Eigenvalue: {top_eigenvalue}")

        from lib.utils.pyhessian.pyhessian import hessian

        hessian_comp = hessian(
            model, loss_fn, dataloader=dataloader["train_loader"], cuda=args.device
        )
        density_eigen, density_weight = hessian_comp.density()

        # print('\n***Top Eigenvalues: ', top_eigenvalues)
        # print('\n***Trace: ', np.mean(trace))

        get_esd_plot(density_eigen, density_weight)
        # TODO: check how many eigenvalues are negative, i.e. negative curvature
        # -> not converged to perfect local minimum that satisfies 1st and 2nd
        # optimality conditions

        if args.hessian_directions:
            max_ev, max_evec, min_ev, min_evec = compute_dominant_hessian_directions(
                model,
                loss_fn,
                dataloader["train_loader"],
                tol=1e-4,  # Pass train_loader
            )
            save_eigenvectors_to_hdf5(
                args=args,
                net=model,
                max_evec=max_evec,
                min_evec=min_evec,
                output_dir=log_dir + "hessian_directions",
            )


if __name__ == "__main__":
    main()

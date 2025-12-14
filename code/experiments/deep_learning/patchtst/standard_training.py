import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# TODO: change paths here, don't need all of em

sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))


# from src.models.time_series.patchtst import PatchTST

from src.models.time_series.patchtst.PatchTST_test import PatchTST
from src.engines.patchtst_engine import PatchTST_Engine
from src.utils.args import get_patchtst_config
from src.utils.dataloader import (
    SamformerDataloader,
)
from src.utils.logging import get_logger

# TODO: write this in a separate class
from src.utils.samformer_utils.sam import SAM

from src.utils.reproducibility import set_seed
from src.utils.experiment_utils import run_experiments_on_dataloader_list
from src.utils.model_utils import load_optimizer

import torch

torch.set_num_threads(3)
from lib.utils.pyhessian.density_plot import get_esd_plot
from lib.optimizers.gsam.gsam.gsam import GSAM
from lib.optimizers.gsam.gsam.scheduler import LinearScheduler

import argparse

torch.set_num_threads(3)
from lib.utils.pyhessian.density_plot import get_esd_plot
from lib.optimizers.gsam.gsam.gsam import GSAM
from lib.optimizers.gsam.gsam.scheduler import LinearScheduler


# args from orig patchtst repo
#
#
# python -u run_longExp.py --random_seed 2021 --is_training 1 --root_path ./dataset/ --data_path ETTh1.csv --model_id 336_96 --model PatchTST --data ETTh1 --features M --seq_len 336 --pred_len 96 --enc_in 7 --e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --des Exp --train_epochs 100 --itr 1 --batch_size 128 --learning_rate 0.0001
# Args in experiment:
# Namespace(random_seed=2021, is_training=1, model_id='336_96', model='PatchTST', data='ETTh1', root_path='./dataset/', data_path='ETTh1.csv', features='M', target='OT', freq='h', checkpoints='./checkpoints/', seq_len=336, label_len=48, pred_len=96, fc_dro
# pout=0.3, head_dropout=0.0, patch_len=16, stride=8, padding_patch='end', revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25, individual=0, embed_type=0, enc_in=7, dec_in=7, c_out=7, d_model=16, n_heads=4, e_layers=3, d_layers=1, d_ff=128
# , moving_avg=25, factor=1, distil=True, dropout=0.3, embed='timeF', activation='gelu', output_attention=False, do_predict=False, num_workers=10, itr=1, train_epochs=100, batch_size=128, patience=100, learning_rate=0.0001, des='Exp', loss='mse', lradj='ty
# pe3', pct_start=0.3, use_amp=False, use_gpu=True, gpu=0, use_multi_gpu=False, devices='0,1,2,3', test_flop=False)
# Use GPU: cuda:0
#
# best:
# python code/experiments/deep_learning/patchtst/standard_training.py --dataset ETTh1 --seed 2021 --device cuda:0 --lrate 1e-4 --patience 5 --max_epoch 300 --rho 0.4 --batch_size 128 --seq_len 512 --horizon 96 --n_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3 --fc_dropout 0.3 --head_dropout 0 --patch_len 16 --stride 8 --use_revin --revin_affine


def get_config():
    parser = get_patchtst_config()
    args = parser.parse_args()
    args._parser = parser

    args.model_name = "patchtst"

    base_dir = SCRIPT_DIR.parents[3] / "results"

    # TODO: change this. Maybe put all possible model results into a dir called
    # "patchtst", then add subdirs for sam, gsam etc.

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

    model = PatchTST(
        # Core architecture parameters
        enc_in=args.enc_in,
        seq_len=args.seq_len,
        pred_len=args.horizon,
        n_layers=args.n_layers,
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
        # RevIN parameters
        revin=args.use_revin,
        affine=args.revin_affine,
        subtract_last=args.revin_subtract_last,
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

    optimizer = load_optimizer(model, args, logger)
    model.print_model_summary(args, logger)

    # Setup learning rate scheduler and SAM if needed
    lr_scheduler = None
    if args.sam:
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

    loss_fn = torch.nn.MSELoss()

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
        sam=args.sam,
        gsam=args.gsam,
        scheduler=lr_scheduler,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        # metrics=["mse", "mape", "rmse"],  # Track these metrics
        metrics=["mse"],  # Track these metrics
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


if __name__ == "__main__":
    main()

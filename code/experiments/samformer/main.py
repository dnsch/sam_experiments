import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "extra" / "loss_landscape"))

from src.models.samformer import SAMFormerArchitecture
from src.engines.samformer_engine import SAMFormer_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import (
    SamformerDataloader,
)
from src.utils.logging import get_logger
from src.utils.samformer_utils.sam import SAM
from src.utils.functions import (
    compute_dominant_hessian_directions,
    save_eigenvectors_to_hdf5,
)

from src.utils.functions import compute_top_eigenvalue_and_eigenvector 

import numpy as np
import torch
import pdb
import copy


import matplotlib.pyplot as plt

torch.set_num_threads(3)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_public_config()

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimizer to use (e.g., Adam, SGD, Adagrad). Use same case as class names in torch.optim",
    )
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    # SAMFormer hyperparameters
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--use_revin", type=bool, default=True)
    parser.add_argument("--num_channels", type=int, default=True)
    parser.add_argument("--hid_dim", type=int, default=16)
    parser.add_argument(
        "--no_sam", action="store_true", help="don't use Sharpness Awarae Minimization"
    )
    args = parser.parse_args()

    if args.model_name == "":
        args.model_name = "samformer"
    if args.dataset == "":
        args.dataset = "ETTh1"
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


def load_optimizer(model, args, logger):
    try:
        optimizer_class = getattr(torch.optim, args.optimizer)

        if not args.no_sam:
            logger.info(f"Optimizer class: {optimizer_class}")
            return optimizer_class

        optimizer = optimizer_class(
            model.parameters(), lr=args.lrate, weight_decay=args.wdecay
        )
        logger.info(optimizer)
        return optimizer
    except AttributeError:
        raise ValueError(f"Optimizer '{args.optimizer}' not found in torch.optim.")


def main():
    args, log_dir, logger = get_config()

    set_seed(args.seed)

    dataset_name = args.dataset
    time_increment = 1
    dataloader_instance = SamformerDataloader(
        dataset_name, args, logger, time_increment
    )
    dataloader = dataloader_instance.get_dataloader()

    model = SAMFormerArchitecture(
        node_num=None,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        num_channels=dataloader["train_loader"].dataset[0][0].shape[0],
        seq_len=args.seq_len,
        hid_dim=args.hid_dim,
        horizon=args.horizon,
        use_revin=args.use_revin,
    )

    optimizer = load_optimizer(model, args, logger)
    if not args.no_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optimizer,
            rho=args.rho,
            lr=args.lrate,
            weight_decay=args.wdecay,
        )

    loss_fn = torch.nn.MSELoss()

    engine = SAMFormer_Engine(
        device=args.device,
        model=model,
        dataloader=dataloader,
        # scaler=scaler,
        scaler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=None,
        clip_grad_value=0,
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
    )

    if args.mode == "train":
        engine.train()
    elif args.mode == "test":
        engine.evaluate(args.mode)


    top_eigenvalue, top_eigenvector = compute_top_eigenvalue_and_eigenvector(model, loss_fn, dataloader["train_loader"])
    print(f"Max Eigenvalue: {top_eigenvalue}")

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

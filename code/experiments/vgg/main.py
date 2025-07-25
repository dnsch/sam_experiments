import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))

from src.models.vgg import VGG9, VGG16, VGG19
from src.engines.vgg_engine import VGG_Engine
from src.utils.args import get_public_config
from src.utils.dataloader import (
    CIFAR10Dataloader,
)
from src.utils.logging import get_logger

from extra.sam.sam import SAM

import numpy as np
import torch

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
    parser.add_argument(
        "--lrate_decay", default=0.1, type=float, help="learning rate decay rate"
    )
    parser.add_argument("--wdecay", type=float, default=1e-5)
    parser.add_argument("--clip_grad_value", type=float, default=0)
    # SAMFormer hyperparameters
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument(
        "--no_sam", action="store_true", help="don't use Sharpness Awarae Minimization"
    )
    args = parser.parse_args()

    # Logger
    if args.model_name == "":
        args.model_name = "vgg"
    if args.dataset == "":
        args.dataset = "CIFAR10"
    base_dir = SCRIPT_DIR.parents[2] / "results"

    # Logger
    log_dir = "{}/{}/{}/seed_{}_bs_{}_rho_{}/".format(
        base_dir,
        args.model_name,
        args.dataset,
        args.seed,
        args.batch_size,
        args.rho,
    )
    # Logger
    if args.no_sam:
        log_dir = "{}/{}/{}/seed_{}_bs_{}_rho_{}/".format(
            base_dir,
            args.model_name + "_without_sam",
            args.dataset,
            args.seed,
            args.seq_len,
            args.horizon,
            args.batch_size,
        )
    else:
        log_dir = "{}/{}/{}/seed_{}_bs_{}_rho_{}/".format(
            base_dir,
            args.model_name + "_sam",
            args.dataset,
            args.seed,
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

    dataloader_instance = CIFAR10Dataloader(
        args,
    )
    dataloader = dataloader_instance.get_dataloader()

    model = VGG9()

    optimizer = load_optimizer(model, args, logger)

    if not args.no_sam:
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optimizer,
            rho=args.rho,
            lr=args.lrate,
            weight_decay=args.wdecay,
        )

    loss_fn = torch.nn.CrossEntropyLoss()

    engine = VGG_Engine(
        device=args.device,
        model=model,
        dataloader=dataloader,
        scaler=None,
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=None,
        clip_grad_value=0,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        batch_size=args.batch_size,
        no_sam=args.no_sam,
    )

    if args.mode == "train":
        engine.train()
    model = engine.return_save_model(engine._save_path)


if __name__ == "__main__":
    main()

import sys

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# TODO: change paths here, don't need all of em
sys.path.append(str(SCRIPT_DIR.parents[1]))
sys.path.append(str(SCRIPT_DIR.parents[2]))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "pyhessian"))
sys.path.append(str(SCRIPT_DIR.parents[2] / "lib" / "utils" / "loss_landscape"))

from src.models.time_series.tsmixer import TSMixer
from src.models.time_series.tsmixer import TSMixerExt
from src.engines.tsmixer_engine import TSMixer_Engine
from src.utils.args import get_tsmixer_config
from src.utils.dataloader import (
    SamformerDataloader,  # Assuming this can be reused or rename to generic dataloader
)
from src.utils.logging import get_logger
from src.utils.samformer_utils.sam import SAM
from src.utils.functions import (
    compute_dominant_hessian_directions,
    save_eigenvectors_to_hdf5,
)

import numpy as np
import torch
import pdb

torch.set_num_threads(3)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_config():
    parser = get_tsmixer_config()
    args = parser.parse_args()

    args.model_name = "tsmixer"
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
    dataloader_instance = (
        SamformerDataloader(  # TODO: change Dataloader to more generic name
            dataset_name,
            args,
            logger,
            time_increment,  # as data originates from tsmixer
        )
    )
    dataloader = dataloader_instance.get_dataloader()

    # TODO: think about deriving this directly from the data for all main.py
    # not add it as an argument
    num_channels = dataloader["train_loader"].dataset[0][0].shape[0]
    args.num_channels = num_channels
    args.input_dim = num_channels
    args.output_dim = num_channels

    model = TSMixer(
        num_channels=args.num_channels,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        seq_len=args.seq_len,
        horizon=args.horizon,
        # TSMixer specific parameters
        activation_fn=args.activation_fn,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        ff_dim=args.ff_dim,
        normalize_before=args.normalize_before,
        norm_type=args.norm_type,
        use_revin=args.use_revin,
    )
    # model = TSMixerExt(
    #     num_channels=args.num_channels,
    #     input_dim=args.input_dim,
    #     output_dim=args.output_dim,
    #     seq_len=args.seq_len,
    #     horizon=args.horizon,
    #     # TSMixer specific parameters
    #     activation_fn=args.activation_fn,
    #     num_blocks=args.num_blocks,
    #     dropout_rate=args.dropout_rate,
    #     ff_dim=args.ff_dim,
    #     normalize_before=args.normalize_before,
    #     norm_type=args.norm_type,
    #     use_revin=args.use_revin,
    # )

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

    engine = TSMixer_Engine(
        device=args.device,
        model=model,
        dataloader=dataloader,
        scaler=None,  # TODO: change base engine and remove scaler as lots of models dont use it
        loss_fn=loss_fn,
        lrate=args.lrate,
        optimizer=optimizer,
        scheduler=None,
        clip_grad_value=args.clip_grad_value,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_dir=log_dir,
        logger=logger,
        seed=args.seed,
        batch_size=args.batch_size,
        num_channels=num_channels,
        pred_len=args.horizon,
        no_sam=args.no_sam,
        use_revin=args.use_revin,
    )

    if args.mode == "train":
        engine.train()
    elif args.mode == "test":
        engine.evaluate(args.mode)

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
            output_dir=log_dir / "hessian_directions",  # Fixed path concatenation
        )


if __name__ == "__main__":
    main()

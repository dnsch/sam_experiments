from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))
from src.engines.samformer_engine import SAMFormer_Engine

# ============================================================

# Experiments

# ============================================================


def get_engine(
    model_name,
    device,
    model,
    dataloader,
    scaler,
    loss_fn,
    optimizer,
    lr_scheduler,
    log_dir,
    logger,
    args,
):
    """
    Function that creates and returns the appropriate engine based on model_name.

    Args:
        model_name: Name of the model (e.g., 'samformer')
        device: Device to run on
        model: The model instance
        dataloader: Dictionary containing train/val/test dataloaders
        scaler: Scaler for data normalization
        loss_fn: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        log_dir: Directory for logging
        logger: Logger object
        args: Arguments object containing hyperparameters

    Returns:
        engine: The appropriate engine instance
    """
    # TODO: use try to get num_channels and issue warning if that's unsuccessful
    if model_name == "samformer":
        engine = SAMFormer_Engine(
            device=device,
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
            use_revin=args.use_revin,
            gsam=args.gsam,
            plot_attention=args.plot_attention,
        )
    # elif model_name.lower() == "transformer":
    #     engine = Transformer_Engine(...)
    # elif model_name.lower() == "lstm":
    #     engine = LSTM_Engine(...)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return engine


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

    Args:
        dataloader_list: List of dictionaries containing train/val/test dataloaders
        args: Arguments object containing hyperparameters and configuration
        model: The model to train/evaluate
        loss_fn: Loss function
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        log_dir: Directory for logging
        logger: Logger object

    Returns:
        results: List of results from each experiment
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

        # TODO: put this in a separate function
        # Create the engine
        engine = get_engine(
            model_name=args.model_name,
            device=args.device,
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_dir=experiment_log_dir,
            logger=logger,
            args=args,
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

import argparse


def get_public_config():
    parser = argparse.ArgumentParser()
    # Hardware
    # parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device", type=str, default="cpu")
    # parser.add_argument("--dataset", type=str, default="")
    # if need to use the data from multiple years, please use underline to separate them, e.g., 2018_2019
    # parser.add_argument('--years', type=str, default='2019')

    # Model
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument(
        "--loss_name",
        "-l",
        default="mse",
        help="loss functions: crossentropy | mse",
    )

    # Dataset
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument(
        "--raw_data", action="store_true", default=False, help="do not normalize data"
    )
    parser.add_argument(
        "--noaug", default=False, action="store_true", help="no data augmentation"
    )
    parser.add_argument("--label_corrupt_prob", type=float, default=0.0)
    # parser.add_argument(
    #     "--trainloader", default="", help="path to the dataloader with random labels"
    # )
    # parser.add_argument(
    #     "--testloader", default="", help="path to the testloader with random labels"
    # )

    # Experiment
    # seq_len denotes input history length, horizon denotes output future length
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    # patience for early stopping
    parser.add_argument("--patience", type=int, default=30)

    # Loss-landscape (https://github.com/tomgoldstein/loss-landscape)
    parser.add_argument(
        "--plot_surface_mpi",
        "-m",
        action="store_true",
        help="use mpi for loss landscape",
    )
    parser.add_argument(
        "--plot_surface_cuda",
        "-c",
        action="store_true",
        help="use cuda for loss landscape",
    )
    parser.add_argument("--threads", default=2, type=int, help="number of threads")
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="number of GPUs to use for each rank, useful for data parallel evaluation",
    )

    # model parameters
    parser.add_argument("--model", default="samformer", help="model name")
    parser.add_argument(
        "--model_file",
        default="experiments/samformer/ETTh1/final_model_s2024.pt",
        help="path to the trained model file",
    )
    parser.add_argument(
        "--model_file2",
        default="",
        help="use (model_file2 - model_file) as the xdirection",
    )
    parser.add_argument(
        "--model_file3",
        default="",
        help="use (model_file3 - model_file) as the ydirection",
    )

    # direction parameters
    parser.add_argument(
        "--dir_file",
        default="",
        help="specify the name of direction file, or the path to an eisting direction file",
    )
    parser.add_argument(
        "--dir_type",
        default="weights",
        help="direction type: weights | states (including BN's running_mean/var)",
    )
    parser.add_argument(
        "--x", default="-1:1:51", help="A string with format xmin:x_max:xnum"
    )
    parser.add_argument("--y", default=None, help="A string with format ymin:ymax:ynum")
    parser.add_argument(
        "--xnorm", default="", help="direction normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--ynorm", default="", help="direction normalization: filter | layer | weight"
    )
    parser.add_argument(
        "--xignore", default="", help="ignore bias and BN parameters: biasbn"
    )
    parser.add_argument(
        "--yignore", default="", help="ignore bias and BN parameters: biasbn"
    )
    parser.add_argument(
        "--same_dir",
        action="store_true",
        default=False,
        help="use the same random direction for both x-axis and y-axis",
    )
    parser.add_argument(
        "--idx", default=0, type=int, help="the index for the repeatness experiment"
    )
    parser.add_argument(
        "--surf_file",
        default="",
        help="customize the name of surface file, could be an existing file.",
    )
    parser.add_argument(
        "--hessian_directions",
        action="store_true",
        default=False,
        help="create hessian eigenvectors directions h5 file",
    )

    # plot parameters
    parser.add_argument(
        "--proj_file",
        default="",
        help="the .h5 file contains projected optimization trajectory.",
    )
    parser.add_argument(
        "--loss_max", default=5, type=float, help="Maximum value to show in 1D plot"
    )
    parser.add_argument("--vmax", default=10, type=float, help="Maximum value to map")
    parser.add_argument("--vmin", default=0.1, type=float, help="Miminum value to map")
    parser.add_argument(
        "--vlevel", default=0.5, type=float, help="plot contours every vlevel"
    )
    parser.add_argument(
        "--show", action="store_true", default=False, help="show plotted figures"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="use log scale for loss values",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="plot figures after computation",
    )

    return parser

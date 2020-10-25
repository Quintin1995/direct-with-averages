# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import torch
import sys
import pathlib
import os

import direct.launch
import functools

from direct.environment import Args
from direct.inference import setup_inference_save_to_h5
from direct.common.subsample import build_masking_function
from direct.utils import set_all_seeds

logger = logging.getLogger(__name__)


def _get_settings(validation_index, env):
    dataset_cfg = env.cfg.validation.datasets[validation_index]
    mask_func = build_masking_function(**dataset_cfg.transforms.masking)
    return dataset_cfg, mask_func


if __name__ == "__main__":
    # This sets MKL threads to 1.
    # DataLoader can otherwise bring a lot of difficulties when computing CPU FFTs in the transforms.
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    # Remove warnings from named tensors being experimental
    os.environ["PYTHONWARNINGS"] = "ignore"

    epilog = f"""
        Examples:
        Run on single machine:
            $ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> [--other-flags]
        Run on multiple machines:
            (machine0)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} data_root output_directory --checkpoint <checkpoint_num> --name <name> --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """

    parser = Args(epilog=epilog)
    parser.add_argument(
        "data_root", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument(
        "output_directory", type=pathlib.Path, help="Path to the output directory."
    )
    parser.add_argument(
        "experiment_directory",
        type=pathlib.Path,
        help="Path to the directory with checkpoints and config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        required=True,
        help="Number of an existing checkpoint.",
    )
    parser.add_argument(
        "--validation-index",
        type=int,
        required=True,
        help="This is the index of the validation set in the config, e.g., 0 will select the first validation set.",
    )
    parser.add_argument(
        "--filenames-filter",
        type=pathlib.Path,
        help="Path to list of filenames to parse.",
    )
    parser.add_argument(
        "--name",
        help="Run name.",
        required=True,
        type=str
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Config file for inference. "
        "Only use it to overwrite the standard loading of the config in the project directory.",
        required=False,
        type=pathlib.Path,
    )

    args = parser.parse_args()
    set_all_seeds(args.seed)

    setup_inference_save_to_h5 = functools.partial(
        setup_inference_save_to_h5, functools.partial(_get_settings, args.validation_index))
    volume_post_processing_func = None
    direct.launch.launch(
        setup_inference_save_to_h5,
        args.num_machines,
        args.num_gpus,
        args.machine_rank,
        args.dist_url,
        args.name,
        args.data_root,
        args.experiment_directory,
        args.output_directory,
        args.filenames_filter,
        args.checkpoint,
        args.device,
        args.num_workers,
        args.machine_rank,
        volume_post_processing_func,
        args.mixed_precision,
        args.debug,
    )

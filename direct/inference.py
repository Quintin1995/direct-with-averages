# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import sys
from functools import partial
from typing import Callable, DefaultDict, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig

from direct.data.datasets import build_dataset_from_input
from direct.data.mri_transforms import build_mri_transforms
from direct.environment import setup_inference_environment
from direct.types import FileOrUrl, PathOrString
from direct.utils import chunks, dict_flatten, remove_keys
from direct.utils.io import read_list
from direct.utils.writers import write_output_to_h5

logger = logging.getLogger(__name__)


def setup_inference_save_to_h5(
    get_inference_settings: Callable,
    run_name: str,
    data_root: Union[PathOrString, None],
    base_directory: PathOrString,
    output_directory: PathOrString,
    filenames_filter: Union[List[PathOrString], None],
    checkpoint: FileOrUrl,
    device: str,
    num_workers: int,
    machine_rank: int,
    cfg_file: Union[PathOrString, None] = None,
    process_per_chunk: Optional[int] = None,
    mixed_precision: bool = False,
    debug: bool = False,
    is_validation: bool = False,
) -> None:
    """This function contains most of the logic in DIRECT required to launch a multi-gpu / multi-node inference process.

    It saves predictions as `.h5` files.

    Parameters
    ----------
    get_inference_settings: Callable
        Callable object to create inference dataset and environment.
    run_name: str
        Experiment run name. Can be an empty string.
    data_root: Union[PathOrString, None]
        Path of the directory of the data if applicable for dataset. Can be None.
    base_directory: PathOrString
        Path to directory where where inference logs will be stored. If `run_name` is not an empty string,
        `base_directory / run_name` will be used.
    output_directory: PathOrString
        Path to directory where output data will be saved.
    filenames_filter: Union[List[PathOrString], None]
        List of filenames to include in the dataset (if applicable). Can be None.
    checkpoint: FileOrUrl
        Checkpoint to a model. This can be a path to a local file or an URL.
    device: str
        Device name.
    num_workers: int
        Number of workers.
    machine_rank: int
        Machine rank.
    cfg_file: Union[PathOrString, None]
        Path to configuration file. If None, will search in `base_directory`.
    process_per_chunk: int
        Processes per chunk number.
    mixed_precision: bool
        If True, mixed precision will be allowed. Default: False.
    debug: bool
        If True, debug information will be displayed. Default: False.
    is_validation: bool
        If True, will use settings (e.g. `batch_size` & `crop`) of `validation` in config.
        Otherwise it will use `inference` settings. Default: False.

    Returns
    -------
    None
    """

    # Step 1 - Setup an inference environment where we can run the inference.
    env = setup_inference_environment(
        run_name, base_directory, device, machine_rank, mixed_precision, cfg_file, debug=debug
    )
    dataset_cfg, transforms = get_inference_settings(env)

    # Trigger cudnn benchmark when the number of different input masks_dict is small.
    torch.backends.cudnn.benchmark = True
    if data_root:
        if filenames_filter:
            filenames_filter = [data_root / _ for _ in read_list(filenames_filter)]

    if not process_per_chunk:
        filenames_filter = [filenames_filter]
    else:
        filenames_filter = list(chunks(filenames_filter, process_per_chunk))

    logger.info(f"Predicting dataset and saving in: {output_directory}.")
    logger.info(f"QVL - The variable is_validation is set to: {is_validation}.")

    if is_validation:
        print(f"QVL1 - Validation batch size: {env.cfg.validation.batch_size}")
        print(f"QVL2 - Validation crop: {env.cfg.validation.crop}")
        batch_size, crop = env.cfg.validation.batch_size, env.cfg.validation.crop  # type: ignore
    else:
        batch_size, crop = env.cfg.inference.batch_size, env.cfg.inference.crop  # type: ignore

    # Step 2 - Actually run the inference to make the reconstructions.
    for curr_filenames_filter in filenames_filter:
        output = inference_on_environment(
            env              = env,
            data_root        = data_root,
            dataset_cfg      = dataset_cfg,
            transforms       = transforms,
            experiment_path  = base_directory / run_name,
            checkpoint       = checkpoint,
            num_workers      = num_workers,
            filenames_filter = curr_filenames_filter,
            batch_size       = batch_size,
            crop             = crop,
        )

        # Step 3 - Write the output to disk.
        modelname = str(env.cfg.model.model_name).split('.')[-1]
        write_output_to_h5(
            output,
            output_directory,
            output_key           = "reconstruction",
            avg_acc              = env.cfg.inference.dataset.avg_acceleration,
            modelname            = modelname,
            also_write_nifti     = True,
            do_round             = False,
            added_gaussian_noise = env.cfg.inference.dataset.add_gaussian_noise,
            db_path              = env.cfg.inference.dataset.db_path,
            tablename            = env.cfg.inference.dataset.tablename,
        )


def build_inference_transforms(env, mask_func: Callable, dataset_cfg: DictConfig) -> Callable:
    """Builds inference transforms."""
    partial_build_mri_transforms = partial(
        build_mri_transforms,
        forward_operator=env.engine.forward_operator,
        backward_operator=env.engine.backward_operator,
        mask_func=mask_func,
    )
    transforms = partial_build_mri_transforms(**dict_flatten(remove_keys(dataset_cfg.transforms, "masking")))
    return transforms


def inference_on_environment(
    env,
    data_root: Union[PathOrString, None],
    dataset_cfg: DictConfig,
    transforms: Callable,
    experiment_path: PathOrString,
    checkpoint: FileOrUrl,
    num_workers: int = 0,
    filenames_filter: Union[List[PathOrString], None] = None,
    batch_size: int = 1,
    crop: Optional[str] = None,
) -> Union[Dict, DefaultDict]:
    """Performs inference on environment.

    Parameters
    ----------
    env: Environment.
    data_root: Union[PathOrString, None]
        Path of the directory of the data if applicable for dataset. Can be None.
    dataset_cfg: DictConfig
        Configuration containing inference dataset settings.
    transforms: Callable
        Dataset transformations object.
    experiment_path: PathOrString
        Path to directory where where inference logs will be stored.
    checkpoint: FileOrUrl
        Checkpoint to a model. This can be a path to a local file or an URL.
    num_workers: int
        Number of workers.
    filenames_filter: Union[List[PathOrString], None]
        List of filenames to include in the dataset (if applicable). Can be None. Default: None.
    batch_size: int
        Inference batch size. Default: 1.
    crop: Optional[str]
        Inference crop type. Can be `header` or None. Default: None.

    Returns
    -------
    output: Union[Dict, DefaultDict]
    """

    logger.warning("pass_h5s and pass_dictionaries is not yet supported for inference.")

    kwargs = {}
    if data_root is not None:
        kwargs.update({"data_root": data_root})
        if filenames_filter:
            kwargs.update({"filenames_filter": filenames_filter})

    dataset = build_dataset_from_input(transforms=transforms, dataset_config=dataset_cfg, **kwargs)

    if len(dataset) <= 0:
        logger.info("Inference dataset is empty. Terminating inference...")
        sys.exit(-1)

    logger.info(f"Inference data size: {len(dataset)}.")

    # Run prediction
    output = env.engine.predict(
        dataset,
        experiment_path,
        checkpoint=checkpoint,
        num_workers=num_workers,
        batch_size=batch_size,
        crop=crop,
    )
    return output

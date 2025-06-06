# coding=utf-8
# Copyright (c) DIRECT Contributors

"""Classes holding the typed configurations for the datasets."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from omegaconf import MISSING

from direct.common.subsample_config import MaskingConfig
from direct.config.defaults import BaseConfig


@dataclass
class CropTransformConfig(BaseConfig):
    crop: Optional[str] = None
    crop_type: Optional[str] = "uniform"
    image_center_crop: bool = False


@dataclass
class SensitivityMapEstimationTransformConfig(BaseConfig):
    estimate_sensitivity_maps: bool = True
    sensitivity_maps_type: str = "rss_estimate"
    sensitivity_maps_espirit_threshold: Optional[float] = 0.05
    sensitivity_maps_espirit_kernel_size: Optional[int] = 6
    sensitivity_maps_espirit_crop: Optional[float] = 0.95
    sensitivity_maps_espirit_max_iters: Optional[int] = 30
    sensitivity_maps_gaussian: Optional[float] = 0.7


@dataclass
class RandomAugmentationTransformsConfig(BaseConfig):
    random_rotation_degrees: Tuple[int, ...] = (-90, 90)
    random_rotation_probability: float = 0.0
    random_flip_type: Optional[str] = "random"
    random_flip_probability: float = 0.0
    random_reverse_probability: float = 0.0


@dataclass
class NormalizationTransformConfig(BaseConfig):
    scaling_key: Optional[str] = "masked_kspace"
    scale_percentile: Optional[float] = 0.99


@dataclass
class TransformsConfig(BaseConfig):
    masking: Optional[MaskingConfig] = MaskingConfig()
    cropping: CropTransformConfig = CropTransformConfig()
    random_augmentations: RandomAugmentationTransformsConfig = RandomAugmentationTransformsConfig()
    padding_eps: float = 0.001
    estimate_body_coil_image: bool = False
    sensitivity_map_estimation: SensitivityMapEstimationTransformConfig = SensitivityMapEstimationTransformConfig()
    normalization: NormalizationTransformConfig = NormalizationTransformConfig()
    delete_acs_mask: bool = True
    delete_kspace: bool = True
    image_recon_type: str = "rss"
    pad_coils: Optional[int] = None
    use_seed: bool = True


@dataclass
class DatasetConfig(BaseConfig):
    name: str = MISSING
    transforms: BaseConfig = TransformsConfig()
    text_description: Optional[str] = None


@dataclass
class H5SliceConfig(DatasetConfig):
    regex_filter: Optional[str] = None
    input_kspace_key: Optional[str] = None
    input_image_key: Optional[str] = None
    kspace_context: int = 0
    pass_mask: bool = False
    data_root: Optional[str] = None
    filenames_filter: Optional[List[str]] = None
    filenames_lists: Optional[List[str]] = None
    filenames_lists_root: Optional[str] = None
    store_applied_acs_mask: bool = False
    avg_collapse_strat: str = "allavg"


@dataclass
class CMRxReconConfig(DatasetConfig):
    regex_filter: Optional[str] = None
    data_root: Optional[str] = None
    filenames_filter: Optional[List[str]] = None
    filenames_lists: Optional[List[str]] = None
    filenames_lists_root: Optional[str] = None
    kspace_key: str = "kspace_full"
    compute_mask: bool = False
    extra_keys: Optional[List[str]] = None
    kspace_context: Optional[str] = None


@dataclass
class FastMRIConfig(H5SliceConfig):
    pass_attrs: bool = True


@dataclass
class FastMRIAvgCombConfig(H5SliceConfig):              # The configuration for the FastMRI dataset with added averages
    pass_attrs: bool                   = True
    avg_acceleration: float            = 1.0            # the acceleration as string based on leaving out averages with added equispaced acceleration symbolising the application of a GRAPPA factor for example
    echo_train_length: Optional[int]   = 25             # The echo train length of the data, typically 25
    add_gaussian_noise: Optional[bool] = False          # Add Gaussian noise to the data
    noise_mult: Optional[float]        = 2.5            # The fraction of noise to add to the data, determined empirically based on the max of the data
    db_path: Optional[str]             = None           # The path to the database to use for the uncertainty quantification
    tablename: Optional[str]           = "patients_uq"  # The tablename to use for the uncertainty quantification`
    do_lxo_for_uq: bool                = True           # If True, apply fold_idx dropout for Uncertainty Quantification
    echo_train_acceleration: int       = 1              # Acceleration factor; 1 means no acceleration by means of droping out a factor of the echo train length
    echo_train_fold_idx: int           = 0              # Index of ET(s) to leave out from retained set
    # leave_out_echo_trains: Optional[int] = 2          # The number of echo trains to leave out for leave-X-out uncertainty quantification
    

@dataclass
class CalgaryCampinasConfig(H5SliceConfig):
    crop_outer_slices: bool = False


@dataclass
class FakeMRIBlobsConfig(DatasetConfig):
    pass_attrs: bool = True


@dataclass
class SheppLoganDatasetConfig(DatasetConfig):
    shape: Tuple[int, int, int] = (100, 100, 30)
    num_coils: int = 12
    seed: Optional[int] = None
    B0: float = 3.0
    zlimits: Tuple[float, float] = (-0.929, 0.929)


@dataclass
class SheppLoganProtonConfig(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT1Config(SheppLoganDatasetConfig):
    pass


@dataclass
class SheppLoganT2Config(SheppLoganDatasetConfig):
    T2_star: bool = False

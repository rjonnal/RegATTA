# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:59:56 2025

@author: ZAQ
"""

# __init__.py

"""
High-level API for the OCT/AO-OCT 3D volume registration package.

Convenience imports:
--------------------
Classes:
    - RegisteredImageSeries
    - BroadcastReference
    - PerImageRegistration
    - RegistrationResult

Functions:
    - register_by_broadcasting
    - build_and_register_from_root
    - load_volume_from_folder
    - auto_crop_volume
    - flatten_volume_x
    - upsample_volume
"""

from .ris_core import RegisteredImageSeries
from .registration_broadcast import (
    BroadcastReference,
    PerImageRegistration,
    RegistrationResult,
    register_by_broadcasting,
)
from .ris_pipeline import build_and_register_from_root
from .io_preprocess import (
    load_volume_from_folder,
    auto_crop_volume,
    flatten_volume_x,
    upsample_volume,
)

__all__ = [
    "RegisteredImageSeries",
    "BroadcastReference",
    "PerImageRegistration",
    "RegistrationResult",
    "register_by_broadcasting",
    "build_and_register_from_root",
    "load_volume_from_folder",
    "auto_crop_volume",
    "flatten_volume_x",
    "upsample_volume",
]

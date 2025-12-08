# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:58:11 2025

@author: ZAQ
"""

# ris_pipeline.py
from __future__ import annotations #automatically detect hints as strings

import glob
import os
from typing import List, Optional, Dict, Any

import numpy as np

from io_preprocess import (
    load_volume_from_folder,
    auto_crop_volume,
    flatten_volume_x,
    upsample_volume,
    default_npy_volume_loader,
)
from registration_broadcast import register_by_broadcasting, RegistrationResult
from ris_core import RegisteredImageSeries


def build_and_register_from_root(
    root: str,
    *,
    prefix: str = "bscan",
    indices: Optional[List[int]] = None,
    crop: bool = False,
    flatten_x: bool = False,
    upsample_factor: Optional[int] = None,
    use_broadcasting: bool = True,
    plot: bool = True,
    save_dir: Optional[str] = None,
    loader=None,
    **broadcast_kwargs,
) -> Dict[str, Any]:
    """
    High-level pipeline:
    1. Discover subfolders under `root`.
    2. Load volumes (optional crop / flatten / upsample).
    3. Register volumes (broadcasting or no-op).
    4. Correct images and return a structured dict.

    Returns
    -------
    dict
        {
          "rvs": RegisteredImageSeries,
          "folders": [...],
          "images": [...],          # possibly cropped/flattened/upsampled
          "registration": RegistrationResult or dict,
          "z_crop": (z1, z2) or None,
        }
    """
    # Discover subfolders
    subfolders = sorted(
        [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    )
    if indices is None:
        indices = list(range(len(subfolders)))
    folders = [subfolders[i] for i in indices]

    if not folders:
        raise ValueError(f"No subfolders found under: {root}")

    vols: List[np.ndarray] = []
    z_crop = None

    # Decide loader
    _loader = loader if loader is not None else load_volume_from_folder

    for folder in folders:
        vol = _loader(folder)  # expected (slow, depth, fast)

        if crop:
            if z_crop is None:
                vol, z_crop = auto_crop_volume(vol)
            else:
                z1, z2 = z_crop
                vol = vol[:, z1:z2, :]

        if flatten_x:
            vol = flatten_volume_x(vol)

        if upsample_factor and upsample_factor > 1:
            vol = upsample_volume(vol, upsample_factor)

        vols.append(vol)

    if use_broadcasting:
        reg: RegistrationResult = register_by_broadcasting(
            ref_image=vols[0],
            target_images=vols,
            plot=plot,
            save_dir=save_dir,
            **broadcast_kwargs,
        )
        rvs = reg.rvs
        registration = reg
    else:
        # No registration; just pack in an RVS and correct (with zero shifts).
        rvs = RegisteredImageSeries()
        for v in vols:
            rvs.add(v)
        rvs.correct_images()
        registration = {
            "rvs": rvs,
            "per_image": [],
            "corrected_images": rvs.corrected_images,
        }

    return {
        "rvs": rvs,
        "folders": folders,
        "images": vols,
        "registration": registration,
        "z_crop": z_crop,
    }

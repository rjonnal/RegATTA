# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:57:24 2025

@author: ZAQ
"""

# io_preprocess.py
from __future__ import annotations #automatically detect hints as strings

import glob
import os
from typing import Callable, Tuple, List

import numpy as np


def _rfunc():
    """
    Lazy import of the helper module. Adjust the import name as needed.
    """
    import importlib

    try:
        return importlib.import_module("registration_functions")
    except Exception as e:
        raise ImportError(
            "ucd_registration_functions.py not found or import failed. "
            "Make sure it is on your PYTHONPATH or next to this file."
        ) from e


# ---------------------------------------------------------------------- #
# Loading & cropping
# ---------------------------------------------------------------------- #

def load_volume_from_folder(folder: str, prefix: str = "bscan") -> np.ndarray:
    """
    Wrapper over rfunc.get_image (returns shape (slow, depth, fast)).
    """
    rfunc = _rfunc()
    vol = rfunc.get_image(folder, prefix=prefix)
    return vol  # (sy, sz, sx)


def auto_crop_volume(vol: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Use rfunc.auto_crop to get z1/z2 based on z-profile, then slice.

    Parameters
    ----------
    vol : ndarray, shape (sy, sz, sx)

    Returns
    -------
    cropped : ndarray, shape (sy, z2-z1, sx)
    z_crop : (z1, z2)
    """
    rfunc = _rfunc()
    z1, z2 = rfunc.auto_crop(vol)
    return vol[:, z1:z2, :], (int(z1), int(z2))


# ---------------------------------------------------------------------- #
# Flattening / upsampling
# ---------------------------------------------------------------------- #

def flatten_volume_x(vol_yzx: np.ndarray) -> np.ndarray:
    """
    Flatten volume in x using rfunc.flatten_image, which expects (slow, fast, depth).
    This utility converts (slow, depth, fast) -> (slow, fast, depth) and back.
    """
    rfunc = _rfunc()
    vol_yxz = np.transpose(vol_yzx, (0, 2, 1))  # (slow, fast, depth)
    flat_yxz = rfunc.flatten_image(vol_yxz)
    flat_yzx = np.transpose(flat_yxz, (0, 2, 1))  # (slow, depth, fast)
    return flat_yzx


def upsample_volume(vol: np.ndarray, factor: int) -> np.ndarray:
    """
    Upsample volume using rfunc.upsample.
    """
    rfunc = _rfunc()
    return rfunc.upsample(vol, factor)


# ---------------------------------------------------------------------- #
# Simple default loader for from_root pipelines
# ---------------------------------------------------------------------- #

def default_npy_volume_loader(folder: str) -> np.ndarray:
    """
    Stack all *.npy b-scans in lexical order into a 3D volume.

    Parameters
    ----------
    folder : str
        Directory containing per-row .npy files.

    Returns
    -------
    vol : ndarray, shape (n_slow, n_depth, n_fast)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {folder}")
    bscans = [np.load(f) for f in files]
    vol = np.stack(bscans, axis=0)
    return vol

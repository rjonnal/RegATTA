# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:57:38 2025

@author: ZAQ
"""

# registration_broadcast.py
from __future__ import annotations #automatically detect hints as strings

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
from matplotlib import pyplot as plt

from ris_core import RegisteredImageSeries


class BroadcastReference:
    """
    Cached-FFT reference volume for broadcasting-based registration.

    Attributes
    ----------
    vol : ndarray, shape (n_slow, n_depth, n_fast)
    fref : ndarray, same shape as `vol`
        Full 3D FFT of the reference volume.
    """

    def __init__(self, vol: np.ndarray) -> None:
        self.vol = vol
        self.fref = np.fft.fftn(vol)
        self.n_slow, self.n_depth, self.n_fast = vol.shape

    @staticmethod
    def _wrap_fix(p: int, size: int) -> int:
        """
        Wrap index on periodic domain [0, size).
        Fix on z, x; not y (to match original behavior).
        """
        return p if p < size // 2 else p - size

    def register_bscan(self, target_bscan: np.ndarray, poxc: bool = True) -> Dict[str, float]:
        """
        Register a single B-scan (depth, fast) to the 3D reference.

        Parameters
        ----------
        target_bscan : ndarray, shape (n_depth, n_fast)
        poxc : bool
            If True, use phase-only cross correlation (normalized).

        Returns
        -------
        dict with keys: dx, dy, dz, xc
        """
        ftar = np.conj(np.fft.fft2(target_bscan))  # (depth, fast)
        prod = self.fref * ftar                    # (slow, depth, fast)

        if poxc:
            prod = prod / (np.abs(prod) + 1e-12)

        xc_arr = np.abs(np.fft.ifftn(prod))        # (slow, depth, fast)

        yp, zp, xp = np.unravel_index(np.argmax(xc_arr), xc_arr.shape)
        sy, sz, sx = xc_arr.shape
        zp = self._wrap_fix(zp, sz)
        xp = self._wrap_fix(xp, sx)

        return dict(dx=int(xp), dy=int(yp), dz=int(zp), xc=float(np.max(xc_arr)))


def _tile_map_from_row_shifts(row_shifts: List[float], n_fast: int) -> np.ndarray:
    """
    Turn a per-row 1D shift list/array into a (n_slow, n_fast) map
    by repeating across the fast axis.
    """
    row_shifts = np.asarray(row_shifts, dtype=float)
    return np.tile(row_shifts[:, None], (1, int(n_fast)))


@dataclass
class PerImageRegistration:
    xc: List[float]
    dx: List[float]
    dy: List[float]
    dz: List[float]
    xmap: np.ndarray
    ymap: np.ndarray
    zmap: np.ndarray


@dataclass
class RegistrationResult:
    rvs: RegisteredImageSeries
    per_image: List[PerImageRegistration]
    corrected_images: List[np.ndarray]


def register_by_broadcasting(
    ref_image: np.ndarray,
    target_images: List[np.ndarray],
    *,
    poxc: bool = True,
    plot: bool = True,
    save_dir: Optional[str] = None,
) -> RegistrationResult:
    """
    Register a list of target volumes to a reference using broadcasting.

    Parameters
    ----------
    ref_image : ndarray, shape (n_slow, n_depth, n_fast)
    target_images : list of ndarray
        Target volumes to register. Typically includes the reference as #0.
    poxc : bool
        Use phase-only cross correlation.
    plot : bool
        If True, plot xc, dy, dz, dx traces per target.
    save_dir : str, optional
        If provided, save xc and shift maps as .npy files.

    Returns
    -------
    RegistrationResult
    """
    rvs = RegisteredImageSeries()
    rvs.add(ref_image)

    bref = BroadcastReference(ref_image)
    sy, sz, sx = ref_image.shape

    per_image_results: List[PerImageRegistration] = []

    for vol_idx, target_vol in enumerate(target_images):
        xcmax_arr: List[float] = []
        x_shift_arr: List[float] = []
        y_shift_arr: List[float] = []
        z_shift_arr: List[float] = []

        for s in range(bref.n_slow):
            bscan = target_vol[s, :, :]  # (depth, fast)
            res = bref.register_bscan(bscan, poxc=poxc)
            xcmax_arr.append(res["xc"])
            x_shift_arr.append(res["dx"])
            y_shift_arr.append(res["dy"] - s)  # keep original y-shift convention
            z_shift_arr.append(res["dz"])

        xmap = _tile_map_from_row_shifts(x_shift_arr, sx)
        ymap = _tile_map_from_row_shifts(y_shift_arr, sx)
        zmap = _tile_map_from_row_shifts(z_shift_arr, sx)

        rvs.add(target_vol, xmap, ymap, zmap)

        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"xcmax_arr_vol{vol_idx}.npy"), np.asarray(xcmax_arr))
            np.save(os.path.join(save_dir, f"x_map_vol{vol_idx}.npy"), xmap)
            np.save(os.path.join(save_dir, f"y_map_vol{vol_idx}.npy"), ymap)
            np.save(os.path.join(save_dir, f"z_map_vol{vol_idx}.npy"), zmap)

        if plot:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(4, 1, 1); ax.plot(xcmax_arr); ax.set_ylabel("xc")
            ax = fig.add_subplot(4, 1, 2); ax.plot(y_shift_arr); ax.set_ylabel("dy")
            ax = fig.add_subplot(4, 1, 3); ax.plot(z_shift_arr); ax.set_ylabel("dz")
            ax = fig.add_subplot(4, 1, 4); ax.plot(x_shift_arr); ax.set_ylabel("dx"); ax.set_xlabel("row")
            fig.suptitle(f"Broadcast registration traces (target vol #{vol_idx})")
            plt.tight_layout()
            plt.show()

        per_image_results.append(
            PerImageRegistration(
                xc=xcmax_arr,
                dx=x_shift_arr,
                dy=y_shift_arr,
                dz=z_shift_arr,
                xmap=xmap,
                ymap=ymap,
                zmap=zmap,
            )
        )

    # Render corrected images via RVS API
    rvs.correct_images()

    return RegistrationResult(
        rvs=rvs,
        per_image=per_image_results,
        corrected_images=rvs.corrected_images,
    )

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:56:17 2025

@author: ZAQ
"""

# ris_core.py
import math
from typing import List, Optional

import numpy as np


class RegisteredImageSeries:
    """
    Container for a series of 3D volumes and their displacement maps
    (x, y, z), with functionality to render motion-corrected images.

    Notes
    -----
    - Volumes are assumed to have shape (slow, depth, fast) = (sy, sz, sx).
    - Shift maps x, y, z are 2D (sy, sx) and are typically integer-valued.
    """

    def __init__(self) -> None:
        self.images: List[np.ndarray] = []
        self.xshifts: List[np.ndarray] = []
        self.yshifts: List[np.ndarray] = []
        self.zshifts: List[np.ndarray] = []
        self.corrected_images: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # Adding images + shift maps
    # ------------------------------------------------------------------ #
    def add(
        self,
        image: np.ndarray,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a 3D image volume and its corresponding displacement fields.

        Parameters
        ----------
        image : ndarray, shape (sy, sz, sx)
            The 3D image volume (slow, depth, fast).
        x, y, z : ndarray, shape (sy, sx), optional
            Per-pixel displacement maps in x-, y-, and z-directions.
            If any is None, zero-valued maps are created.
        """
        if x is None or y is None or z is None:
            sy, sz, sx = image.shape
            x = np.zeros((sy, sx), dtype=float)
            y = np.zeros((sy, sx), dtype=float)
            z = np.zeros((sy, sx), dtype=float)

        self.images.append(image)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)

    # ------------------------------------------------------------------ #
    # Render corrected images on expanded grid
    # ------------------------------------------------------------------ #
    def correct_images(self) -> None:
        """
        Apply per-pixel x, y, and z shift maps to all stored 3D volumes and
        generate motion-corrected (registered) volumes on an expanded grid.

        Steps
        -----
        1. Convert the x/y/z shift lists to NumPy arrays.
        2. Subtract the global minimum from each shift array so all indices
           are non-negative.
        3. Compute expanded dimensions based on the maximum shifts.
        4. For each volume, allocate an expanded complex-valued array filled
           with NaN, and place each A-scan at its shifted location.
        5. Store each corrected volume in `self.corrected_images`.

        Returns
        -------
        None
        """
        if not self.images:
            raise RuntimeError("No images stored in RegisteredImageSeries.")

        sy, sz, sx = self.images[0].shape
        self.corrected_images = []

        # Convert to arrays (shape: n_images, sy, sx)
        x_arr = np.array(self.xshifts, dtype=float)
        y_arr = np.array(self.yshifts, dtype=float)
        z_arr = np.array(self.zshifts, dtype=float)

        # Global min-subtraction to make indices non-negative
        x_arr -= np.min(x_arr)
        y_arr -= np.min(y_arr)
        z_arr -= np.min(z_arr)

        # Save back (if you still want them as attributes)
        self.xshifts = x_arr
        self.yshifts = y_arr
        self.zshifts = z_arr

        expanded_x = sx + math.ceil(np.max(x_arr))
        expanded_y = sy + math.ceil(np.max(y_arr))
        expanded_z = sz + math.ceil(np.max(z_arr))

        for idx, (vol, x_shift_map, y_shift_map, z_shift_map) in enumerate(
            zip(self.images, x_arr, y_arr, z_arr)
        ):
            print(f"Correcting image {idx}")
            print(f"xshifts: {x_shift_map}\nyshifts: {y_shift_map}\nzshifts: {z_shift_map}")

            corrected = np.full(
                (expanded_y, expanded_z, expanded_x),
                np.nan,
                dtype=complex,
            )

            # Place each A-scan at shifted coordinates
            for y in range(sy):
                for x in range(sx):
                    yy = y + int(y_shift_map[y, x])
                    xx = x + int(x_shift_map[y, x])
                    zz = int(z_shift_map[y, x])

                    ascan = vol[y, :, x]  # shape (sz,)
                    corrected[yy, zz:zz + sz, xx] = ascan

            self.corrected_images.append(corrected)

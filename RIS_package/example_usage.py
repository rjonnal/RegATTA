# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:58:33 2025

@author: ZAQ
"""

from ris_pipeline import build_and_register_from_root

result = build_and_register_from_root(
    root=r"E:\your\dataset\root",
    prefix="bscan",
    crop=True,
    flatten_x=True,
    upsample_factor=2,
    use_broadcasting=True,
    plot=True,
    save_dir=r"E:\your\output\registration",
)

rvs = result["rvs"]
corrected_volumes = result["rvs"].corrected_images

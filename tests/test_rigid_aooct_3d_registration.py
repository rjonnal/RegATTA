"""
Created on 03 February 2026
@author:RSJ

In this test we profile 3D volumetric registration.
"""
import regatta
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import cProfile

from regatta import load_image_aooct as load

reference_index = 0
target_index = 3
data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg/'

volume_locations = sorted(glob.glob(os.path.join(data_root,'*')))
reference_location = volume_locations[reference_index]
target_location = volume_locations[target_index]

ref_data = load(reference_location)
tar_data = load(target_location)

def reg3d():
    nxc = np.real(np.fft.ifftn(np.fft.fftn(tar_data)*np.conj(np.fft.fftn(ref_data))))
    y,z,x = np.unravel_index(np.argmax(nxc),nxc.shape)
    return y,z,x

cProfile.run('reg3d()')
print(reg3d())
print(ref_data.shape)

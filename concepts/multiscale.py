"""
Created on 29 January 2026
@author:RSJ

In this test we test the `register` method of `reference_image.ReferenceImage`,
using real AO-OCT small volumes.
"""
import regatta
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob


from regatta import load_image_aooct as load

reference_index = 0
target_index = 0
force_random = True
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct/'
data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_synthetic/'

volume_locations = sorted(glob.glob(os.path.join(data_root,'*')))
reference_location = volume_locations[reference_index]

ref_data = load(reference_location)
target_volume = load(volume_locations[target_index])

if force_random:
    ref_data = np.random.rand(40,100,60)
if force_random:
    target_volume = ref_data


a = np.random.rand(3,3)
b = np.zeros(a.shape)
b[:] = a[:]

a = (a-np.mean(a))/np.std(a)
b = (b-np.mean(b))/np.std(b)


print(np.real(np.fft.ifftn(np.fft.fftn(a)*np.conj(np.fft.fftn(b))))/a.size)

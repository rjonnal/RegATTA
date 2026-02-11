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
target_index = 1
force_random = True

data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg/'
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct/'
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_synthetic/'

volume_locations = sorted(glob.glob(os.path.join(data_root,'*')))
reference_location = volume_locations[reference_index]

ref_data = load(reference_location)
tar_data = load(volume_locations[target_index])


def flythrough(data):
    a_data = 20*np.log10(np.abs(data))

    sy,sz,sx = a_data.shape

    cmin,cmax = np.percentile(a_data,(30,99.9))
    
    for z in range(sz):
        plt.cla()
        plt.imshow(a_data[:,z,:],clim=(cmin,cmax),cmap='gray')
        plt.title('%0.1f dB - %0.1f dB'%(cmin,cmax))
        plt.pause(.1)

class RegisteredImageSeries:

    def __init__(self,reference_data):
        self.images = []
        self.images.append((reference_data,(0,0,0,1.0)))
        self.fref = np.conj(np.fft.fftn(reference_data))
        self.machine_epsilon = np.finfo(float).eps
        
    def add(self,target_data,poxc=True):
        sy,sz,sx = target_data.shape
        
        ftar = np.fft.fftn(target_data)
        prod = ftar*self.fref
        if poxc:
            prod = prod / (np.abs(prod) + self.machine_epsilon)
            
        nxc = np.real(np.fft.ifftn(prod))
        py,pz,px = np.unravel_index(np.argmax(nxc),nxc.shape)
        py = self.wrap_fix(py,sy)
        pz = self.wrap_fix(pz,sz)
        px = self.wrap_fix(px,sx)
        
        correlation = nxc[py,pz,px]
        registration_info = (py,pz,px,correlation)
        self.images.append((target_data,registration_info))
        print(registration_info)

    def wrap_fix(self, p, size):
        # identical wrap convention used in testing_broadcasting.py (fix on z,x; not y)
        return p if p < size // 2 else p - size
        
ris = RegisteredImageSeries(ref_data)
ris.add(tar_data)

#flythrough(ref_data)


"""
Created on 29 January 2026
@author:RSJ

In this test we test the `register` method of `reference_image.ReferenceImage`,
using real AO-OCT small volumes.
"""
import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series


root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct/'

volume_filenames = sorted(glob.glob(os.path.join(root,'*')))

vols = [rfunc.get_volume(fn,prefix='') for fn in volume_filenames]

refidx = len(vols)//2
ref = vols[refidx]
n_slow,n_depth,n_fast = ref.shape
reference_data = ref

rvs = registered_volume_series.RegisteredVolumeSeries(ref)
rvs.register_volumes(vols)

#rfunc.project3(rvs.average_volume,pfunc=np.nanmax)
#rfunc.project3multiple((np.abs(reference_data),rvs.average_volume),pfunc=np.nanmax)
rfunc.project3multiple((np.abs(reference_data),rvs.average_volume),pfunc=rfunc.brightest,clim=(60,100))
#rfunc.flythrough3(rvs.average_volume)




reference_index = 0
target_index = 0
force_random = True
#data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct/'
data_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_synthetic/'

volume_locations = sorted(glob.glob(os.path.join(data_root,'*')))
reference_location = volume_locations[reference_index]

ref_data = load(reference_location)
if force_random:
    ref_data = np.random.rand(40,100,60)


# create a ReferenceImage object based on the sample data
refim = regatta.reference_image.ReferenceImage(ref_data)

target_volume = load(volume_locations[target_index])
if force_random:
    target_volume = ref_data

# Case 0: register 3D ref_data to itself
res = refim.register(ref_data)
try:
    assert res['d0']==0 and res['d1']==0 and res['d2']==0
    print('Test case 0 passed. Registering a volume to itself.')
except AssertionError as ae:
    sys.exit('Test case 0 failed: %s'%res)

# Case 1: register a sequential volume to the reference in 3D
# This case ignores motion warp and assumes there's enough commonality
# even in relatively warped images that an XC peak is apparent.

res = refim.register(target_volume)
try:
    print('Test case 1 passed. Registering two sequential AO-OCT volumes.')
    print('Result: '+'%s'%res)
except AssertionError as ae:
    sys.exit('Test case 1 failed: %s'%res)
    
# Case 2: register B-scans from a target volume to the reference volume

sy,sz,sx = target_volume.shape
for target_slow_coordinate in range(sy):
    target_bscan = target_volume[target_slow_coordinate,:,:]
    res = refim.register(target_bscan)
    xc_arr = res['xc_arr']
    xc_arr_bs = np.fft.fftshift(xc_arr,axes=[1,2])
    d0 = res['d0']
    d1 = res['d1']
    d2 = res['d2']

    d0 = sy-d0-1

    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(xc_arr_bs[d0,:,:])
    plt.colorbar()
    #plt.plot(res['d1'],res['d2'],'rs')
    plt.subplot(2,3,2)
    plt.imshow(np.abs(ref_data[d0,:,:]),cmap='gray')
    plt.subplot(2,3,3)
    plt.imshow(np.abs(target_bscan),cmap='gray')
    plt.subplot(2,3,4)
    plt.imshow(np.max(np.abs(xc_arr_bs),axis=0))
    plt.title('y max proj')
    plt.subplot(2,3,5)
    plt.imshow(np.max(np.abs(xc_arr_bs),axis=1))
    plt.title('z max proj')
    plt.subplot(2,3,6)
    plt.imshow(np.max(np.abs(xc_arr_bs),axis=2))
    plt.title('x max proj')

    plt.suptitle('dy=%d, dz=%d, dx=%d'%(d0,d1,d2))

    
    plt.pause(1)
    plt.show()
    #print(res)

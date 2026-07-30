import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series
import imageio


# data for this test can be downloaded from the link below:
# https://www.dropbox.com/scl/fo/ys0tmdk79i6tvkbz28ktp/ALx_5UovCD0_5O_yz5HGTHw?rlkey=h1g2u6ocv9i00g1o7k9s5lngh&dl=0

# point 'root' to the location of the data downloaded above
root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct'
# root = 'C:/bscans_aooct'

volume_filenames = sorted(glob.glob(os.path.join(root,'*')))[7:13]

vols = [rfunc.get_volume(fn,prefix='') for fn in volume_filenames]

# use a volume from the middle of the series as a reference:
refidx = len(vols)//2

# get the reference volume
ref = vols[refidx]
reference_data = ref

# register the series to the reference:
rvs = registered_volume_series.RegisteredVolumeSeries(ref)
rvs.register_volumes(vols)

# phase-align the volumes
rvs.phase_align_volumes()

# visualize the result somehow:

#rfunc.project3(rvs.average_volume,pfunc=np.nanmax)
rfunc.project3multiple((np.abs(reference_data),rvs.average_volume),pfunc=rfunc.brightest,clim=(60,100))
#rfunc.flythrough3(rvs.average_volume)

np.save('average_volume_ref_%03d.npy'%refidx,rvs.average_volume)

# save the registered, averaged frames as PNG files
temp = np.zeros(rvs.average_volume.shape)
temp[...] = rvs.average_volume[...]
temp = np.abs(temp)
temp = np.round((temp-np.min(temp))/(np.max(temp)-np.min(temp))*255).astype(np.uint8)

png_folder = 'png_ref_%03d'%refidx

os.makedirs(png_folder,exist_ok=True)
for k in range(temp.shape[0]):
    outfn = os.path.join(png_folder,'%05d.png'%k)
    imageio.imwrite(outfn,temp[k,:,:])
    print('Wrote %s.'%outfn)

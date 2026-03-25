import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series

# data for this test can be downloaded from the link below:
# https://www.dropbox.com/scl/fo/ys0tmdk79i6tvkbz28ktp/ALx_5UovCD0_5O_yz5HGTHw?rlkey=h1g2u6ocv9i00g1o7k9s5lngh&dl=0

# point 'root' to the location of the data downloaded above
root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct'
# root = 'C:/aooct_data'

volume_filenames = sorted(glob.glob(os.path.join(root,'*')))

vols = [rfunc.get_volume(fn,prefix='') for fn in volume_filenames]

# use a volume from the middle of the series as a reference:
refidx = len(vols)//2

# get the reference volume
ref = vols[refidx]
reference_data = ref

# register the series to the reference:
rvs = registered_volume_series.RegisteredVolumeSeries(ref)
rvs.register_volumes(vols)

# visualize the result somehow:

#rfunc.project3(rvs.average_volume,pfunc=np.nanmax)
rfunc.project3multiple((np.abs(reference_data),rvs.average_volume),pfunc=rfunc.brightest,clim=(60,100))
#rfunc.flythrough3(rvs.average_volume)

import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series


#root = 'bscans_synthetic'
root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_1deg_corrected'

#uncorrected_root = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_oct_widefield_CHOIR_0052'
#rfunc.reconcile_volume_sizes(uncorrected_root)
#root = uncorrected_root + '_corrected'


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

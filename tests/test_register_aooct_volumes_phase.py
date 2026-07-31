import regatta.registration_functions as rfunc
import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
from regatta import registered_volume_series
import imageio


# data for this test can be downloaded from the link below:
# https://www.dropbox.com/scl/fo/ys0tmdk79i6tvkbz28ktp/ALx_5UovCD0_5O_yz5HGTHw?rlkey=h1g2u6ocv9i00g1o7k9s5lngh&dl=0

# point 'root_folder' to the location of the data downloaded above
root_folder = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct'
output_folder = '/home/rjonnal/Dropbox/Data/volume_registration/bscans_aooct_output'

# root_folder = 'C:/bscans_aooct'

volume_filenames = sorted(glob.glob(os.path.join(root_folder,'*')))

vols = [rfunc.get_volume(fn,prefix='') for fn in volume_filenames]

# use a volume from the middle of the series as a reference:
refidx = len(vols)//2

output_folder = os.path.join(output_folder,'refidx_%05d'%refidx)
os.makedirs(output_folder,exist_ok=True)

# get the reference volume
ref = vols[refidx]
reference_data = ref

# register the series to the reference:
rvs = registered_volume_series.RegisteredVolumeSeries(ref)
rvs.register_volumes(vols)

pre_corrs = []
for k in range(len(vols)):
    pre_corrs.append(rvs.compute_correlation(vols[refidx],vols[k]))

post_corrs = []
for k in range(len(vols)):
    post_corrs.append(rvs.compute_correlation(rvs.corrected_volumes[refidx],rvs.corrected_volumes[k]))

plt.plot(range(len(vols)),pre_corrs,'gs',label='before registration')
plt.plot(range(len(vols)),post_corrs,'ro',label='after registration')
plt.xlabel('volume index')
plt.ylabel('Pearson correlation with volume %d (OCT magnitude only)'%refidx)
plt.legend()
plt.xticks(range(0,20,5))
plt.savefig('volume_correlation.png',dpi=300)
plt.show()
sys.exit()

# phase-align the volumes
rvs.phase_align_volumes()

# visualize the result somehow:

#rfunc.project3(rvs.average_volume,pfunc=np.nanmax)
rfunc.project3multiple((np.abs(reference_data),rvs.average_volume),pfunc=rfunc.brightest,clim=(60,100))
#rfunc.flythrough3(rvs.average_volume)

np.save(os.path.join(output_folder,'average_volume.npy'),rvs.average_volume)

# save the registered, averaged frames as PNG files
temp = np.zeros(rvs.average_volume.shape)
temp[...] = rvs.average_volume[...]
temp = np.abs(temp)
temp = np.round((temp-np.min(temp))/(np.max(temp)-np.min(temp))*255).astype(np.uint8)

png_folder = os.path.join(output_folder,'average_volume_png')
os.makedirs(png_folder,exist_ok=True)

for k in range(temp.shape[0]):
    outfn = os.path.join(png_folder,'%05d.png'%k)
    imageio.imwrite(outfn,temp[k,:,:])
    print('Wrote %s.'%outfn)

# save the corrected volumes:
registered_folder = os.path.join(output_folder,'registered_volumes')
os.makedirs(registered_folder,exist_ok=True)

for idx,vol in enumerate(rvs.corrected_volumes):
    volume_filename = os.path.join(registered_folder,'%05d_registered.npy'%idx)
    np.save(volume_filename,vol)
    print('Saving registered volume %s.'%volume_filename)
        

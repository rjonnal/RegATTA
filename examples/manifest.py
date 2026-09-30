import glob, os

output_folder = '/home/rjonnal/Dropbox/Data/volume_registration/temp/output'

volume_terms = ['/home/rjonnal/Dropbox/Data/volume_registration/temp/07242026_RJ_OD_t1b_acq5/tacq5_bscans/*',
                '/home/rjonnal/Dropbox/Data/volume_registration/temp/07242026_RJ_OD_t3r_acq8/tacq8_bscans/*']

volume_folders = []

for vt in volume_terms:
    volume_folders = volume_folders + glob.glob(vt)

volume_folders = sorted(volume_folders)
reference_folder = volume_folders[0]


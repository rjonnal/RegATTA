import numpy as np
from matplotlib import pyplot as plt
import sys,os,glob
import scipy.interpolate as spi
import scipy.signal as sps
#import inspect
#from matplotlib.widgets import Button, Slider, SpanSelector, Cursor
import pandas as pd
import json

def load_dict(fn):
    with open(fn,'r') as fid:
        s = fid.read()
        d = json.loads(s)
    return d

def save_dict(fn,d):
    s = json.dumps(d)
    with open(fn,'w') as fid:
        fid.write(s)

def generate_registration_manifest(folder, reference_label, upsample_factor = 2):
    outfn = folder + '_registration_manifest.json'
    flist = glob.glob(os.path.join(folder,'*'))
    flist.sort()
    d = {}
    d['reference'] = reference_label
    d['targets'] = flist
    d['upsample_factor'] = upsample_factor
    save_dict(outfn,d)

def get_peaks(prof,count=np.inf):
    # return the COUNT brightest maxima in prof
    left = prof[:-2]
    center = prof[1:-1]
    right = prof[2:]
    peaks = np.where((center>=left)*(center>=right))[0]+1
    if peaks.shape[0] < count:
        print('only %s peaks exists, decrease count to %s'%(peaks.shape[0],peaks.shape[0]))
        count = peaks.shape[0]
    peak_vals = prof[peaks]
    thresh = sorted(peak_vals)[-count]
    peaks = peaks[np.where(prof[peaks]>=thresh)]
    return list(peaks)

def get_volume(folder,prefix='bscan'):
    flist = glob.glob(os.path.join(folder,'%s*.npy'%prefix))
    flist.sort()
    
    vol = [np.load(f) for f in flist]
    vol = np.array(vol)
    
    return vol#[:10,:20,:30]

def upsample(vol,factor):
    return vol.repeat(factor,axis=0).repeat(factor,axis=1).repeat(factor,axis=2)

def nxc3(a,b):
    prod = np.fft.fftn(a)*np.conj(np.fft.fftn(b))
    aprod = np.abs(prod)+1e-16
    #prod = prod/aprod
    out = np.abs(np.fft.ifftn(prod))
    return out

def flythrough3(a,fps=5):
    for k in range(len(a.shape)):
        for d in range(a.shape[0]):
            frame = a[d,:,:]
            if np.all(np.isnan(frame)):
                continue
            plt.cla()
            plt.imshow(a[d,:,:])
            plt.title('dim %d frame %d'%(k,d))
            plt.pause(1.0/fps)
        a = np.transpose(a,[1,2,0])
        

def project3(a,pfunc=np.nanmean):
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1)
    plt.imshow(pfunc(a,axis=0),cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(pfunc(a,axis=1),cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(pfunc(a,axis=2),cmap='gray')
    plt.show()



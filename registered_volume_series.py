import numpy as np
import math
from matplotlib import pyplot as plt
import sys,os,glob

class ReferenceVolume:
    """A class representing the reference volume. This is handy because it
    can store the 3D FFT of the volume such that it doesn't have to be recomputed
    for each B-scan in the target volume."""
    
    def __init__(self,vol):
        self.vol = vol
        self.fref = np.fft.fftn(vol)
        self.n_slow,self.n_depth,self.n_fast = self.vol.shape
        self.rhash = self.get_hash(vol)
        self.cache_folder = './.reference_volume_cache'
        os.makedirs(self.cache_folder,exist_ok=True)
        
    def register(self,target_bscan,poxc=True):
        """Register a single bscan to the reference volume via broadcasting"""
        thash = self.get_hash(target_bscan)
        cachefn = os.path.join(self.cache_folder,'%d_%d.npy'%(self.rhash,thash))
        try:
            xp,yp,zp,xc = np.loadtxt(cachefn)
        except FileNotFoundError:
            ftar = np.conj(np.fft.fft2(target_bscan))
            prod = self.fref*ftar
            if poxc:
                prod = prod/np.abs(prod)
            xc_arr = np.abs(np.fft.ifftn(self.fref*ftar))

            # plt.figure()
            # for k in range(3):
            #     plt.subplot(1,3,k+1)
            #     plt.imshow(np.max(xc_arr,axis=k))
            # plt.show()
            xc = np.max(xc_arr)

            yp,zp,xp = np.unravel_index(np.argmax(xc_arr),xc_arr.shape)
            sy,sz,sx = xc_arr.shape
            #yp = fix(yp,sy)
            zp = self.fix(zp,sz)
            xp = self.fix(xp,sx)
            np.savetxt(cachefn,[xp,yp,zp,xc])
        result = {}
        result['dx'] = xp
        result['dy'] = yp
        result['dz'] = zp
        result['xc'] = xc
        return result

    def fix(self,p,s):
        if p<s//2:
            return p
        else:
            return p-s

    def get_hash(self,vol,N=128):
        return hash(tuple(vol.ravel()[:N]))


class RegisteredVolumeSeries:

    def __init__(self,reference_data):
        self.volumes = []
        self.xshifts = []
        self.yshifts = []
        self.zshifts = []
        self.corrs = []
        self.reference_data = reference_data
        self.ref = ReferenceVolume(reference_data)
        self.add(reference_data)
        
    def add(self,volume,x=None,y=None,z=None,xc=None):

        if x is None or y is None or z is None:
            sy,sz,sx = volume.shape
            x = np.zeros(sy)
            y = np.zeros(sy)
            z = np.zeros(sy)
            xc = np.ones(sy)*np.inf
            
        self.volumes.append(volume)
        self.xshifts.append(x)
        self.yshifts.append(y)
        self.zshifts.append(z)
        self.corrs.append(xc)


    def register_volumes(self,target_volumes_list):

        for vidx,target_volume in enumerate(target_volumes_list):
            corr_arr = []
            y_shift_arr = []
            z_shift_arr = []
            x_shift_arr = []
            for s in range(self.ref.n_slow):
                print(vidx,s)
                tar = target_volume[s,:,:]
                res = self.ref.register(tar)
                corr_arr.append(res['xc'])
                x_shift_arr.append(res['dx'])
                y_shift_arr.append(res['dy']-s)
                z_shift_arr.append(res['dz'])

            # filter shifts using xc here, later

            # ideas for filtering the shift vectors
            # 1. Limit derivative of shift to small amount; motivated
            #    by intuition that the retina's motion between B-scans
            #    is small, esp. at 5 kHz in AO-OCT system
            # 2. Median filtering may remove outliers when they are
            #    isolated, but will fail when clusters of B-scans are out
            #    of place, which appears to happen fequently (during saccades,
            #    for instance)
            # 3. Motion is a non-Markov process; the previous derivative (velocity)
            #    is predictive of the current derivative. This is a stronger statement
            #    than that made in #1 above.
            # 4. Eye movement processes are reversible; that is "previous" in #3 is
            #    equivalent to "next".

            corr_arr = np.array(corr_arr)
            y_shift_arr = np.array(y_shift_arr,dtype=float)
            z_shift_arr = np.array(z_shift_arr,dtype=float)
            x_shift_arr = np.array(x_shift_arr,dtype=float)
            rad_arr = np.sqrt(y_shift_arr**2+z_shift_arr**2+x_shift_arr**2)


            cstd = np.std(corr_arr)
            if cstd>0:
                ncorr_arr = corr_arr/np.std(corr_arr)
            else:
                ncorr_arr = 3.0*np.ones(corr_arr.shape)

            nrad_arr = rad_arr/(np.std(rad_arr)+1)

            ncorr_thresh_std = 1.5
            nrad_thresh_std = 1.0
            valid = (ncorr_arr>ncorr_thresh_std)*(nrad_arr<nrad_thresh_std)
            invalid = 1-valid
            
            corr_arr[np.where(invalid)] = np.nan
            y_shift_arr[np.where(invalid)] = np.nan
            z_shift_arr[np.where(invalid)] = np.nan
            x_shift_arr[np.where(invalid)] = np.nan

            plt.cla()
            plt.plot(ncorr_arr,label='normalized correlation')
            plt.plot(nrad_arr,label='normalized 3D displacement')
            for k in range(len(invalid)):
                if invalid[k]:
                    plt.axvline(k,color='r',alpha=0.2)
            plt.legend()
            os.makedirs('figures',exist_ok=True)
            plt.savefig(os.path.join('figures','regdata_%03d.png'%vidx))
            plt.pause(.1)
            self.add(target_volume,x_shift_arr,y_shift_arr,z_shift_arr,corr_arr)

        self.correct_volumes()
        self.average_volume = np.nanmean(np.abs(np.array(self.corrected_volumes)),axis=0)
        
        
    def correct_volumes(self):
        # Determine how large the rendered volume must be.
        # 1. Convert shifts lists to arrays
        # 2. Subtract minimum from each shift array
        # 3. Compute maximum value of shift in each dimension,
        #    and add this to the original volume dimensions

        sy, sz, sx = self.volumes[0].shape
        self.corrected_volumes = []
        
        xshifts_vec = np.array(self.xshifts)
        yshifts_vec = np.array(self.yshifts)
        zshifts_vec = np.array(self.zshifts)
        corrs_vec = np.array(self.corrs)
        
        # min-subtract all shifts
        yshifts_vec = yshifts_vec - np.nanmin(yshifts_vec)
        zshifts_vec = zshifts_vec - np.nanmin(zshifts_vec)
        xshifts_vec = xshifts_vec - np.nanmin(xshifts_vec)
        
        # corrected dimensions
        csy = int(sy + np.nanmax(yshifts_vec))
        csz = int(sz + np.nanmax(zshifts_vec))
        csx = int(sx + np.nanmax(xshifts_vec))

        for v,ysv,zsv,xsv,cv in zip(self.volumes,yshifts_vec,zshifts_vec,xshifts_vec,corrs_vec):
            corrected_volume = np.zeros((csy,csz,csx),dtype=complex)
            for idx,(bscan,y,z,x,c) in enumerate(zip(v,ysv,zsv,xsv,cv)):
                if np.isnan(c):
                    continue
                corrected_volume[idx+int(y),int(z):int(z+sz),int(x):int(x+sx)] = bscan
                
            self.corrected_volumes.append(corrected_volume)


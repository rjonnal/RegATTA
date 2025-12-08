# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:59:28 2025

@author: ZAQ
"""
import numpy as _np
from matplotlib import pyplot as _plt
import os
import glob as _glob

#%% Synthetic data simulator
# =========================
# Synthetic data Simulator
# =========================

@staticmethod
def simulate_synthetic_data(
    output_root='bscans_synthetic',
    info_dir='bscans_synthetic_info',
    N_scatterers=10000,
    sz_m=500e-6, sy_m=5e-4, sx_m=5e-4,
    L1=1e-6, L2=1.1e-6, N=256,
    lateral_resolution_sigma=5e-6,
    dt=1e-5,                      # 100 kHz, matches original
    x_scan_start=200e-6, x_scan_stop=300e-6, x_step=2e-6,
    y_scan_start=200e-6, y_scan_stop=300e-6, y_step=2e-6,
    n_images=2,
    motion_sigma_xyz=(3e-7, 3e-7, 3e-7),
    seed=None,
    save_png=True
):
    """
    Recreates generate_synthetic_data.py behavior and file outputs:
      - Writes <output_root>/%05d/bscan_%05d.npy B-scan tiles (complex)
      - Writes motion & scan arrays to <info_dir>/...npy
      - Optionally writes PNG previews (same colormap/limits)
    """
    # ---- embed minimal local types to avoid changing external imports ----
    if seed is not None:
        _np.random.seed(seed)

    # (Copied/trimmed structure from original script)  :contentReference[oaicite:3]{index=3}
    class _Scatterer:
        def __init__(self,z,y,x,r): self.z=z; self.y=y; self.x=x; self.r=r
        def lateral_distance(self,y,x): return _np.sqrt((self.x-x)**2+(self.y-y)**2)

    class _Sample:
        def __init__(self, sz_m, sy_m, sx_m):
            self.sz_m=sz_m; self.sy_m=sy_m; self.sx_m=sx_m; self.scatterers=[]
        def add_scatterer(self,z,y,x,r): self.scatterers.append(_Scatterer(z,y,x,r))
        def add_random_scatterer(self):
            self.add_scatterer(_np.random.rand()*self.sz_m,
                               _np.random.rand()*self.sy_m,
                               _np.random.rand()*self.sx_m,
                               _np.random.rand())
        def randomize(self, N): [self.add_random_scatterer() for _ in range(N)]
        def move(self,dz,dy,dx):
            for s in self.scatterers: s.x+=dx; s.y+=dy; s.z+=dz
        def get_visible_scatterers(self,y,x,lim):
            return [s for s in self.scatterers if s.lateral_distance(y,x)<lim]
        
    class _OCT:
        def __init__(self):
            self.N = N
            self.L1 = L1; self.L2 = L2
            self.k1 = 2*_np.pi/self.L2; self.k2 = 2*_np.pi/self.L1
            self.k_arr = _np.linspace(self.k1,self.k2,self.N)
            self.k0 = (self.k1+self.k2)/2
            k_sigma = 1e5
            self.S = _np.exp(-(self.k_arr-self.k0)**2/(2*k_sigma**2))
            self.lateral_resolution_sigma = lateral_resolution_sigma
            self.r_r = 1.0
            self.simplified = True  # match original

        def spectral_scan(self, sample, y, x):
            vis = sample.get_visible_scatterers(y,x,6*self.lateral_resolution_sigma)
            signal = _np.zeros(self.N)
            if not self.simplified:
                signal = signal + self.S*self.r_r
            for s in vis:
                w = _np.exp(-((s.x-x)**2/(2*self.lateral_resolution_sigma**2)+
                              (s.y-y)**2/(2*self.lateral_resolution_sigma**2)))
                if not self.simplified:
                    signal = signal + w*self.S*s.r
                signal = signal + w*2*self.S*_np.sqrt(self.r_r*s.r)*_np.cos(2*self.k_arr*s.z)
            return signal

    # --- instantiate and randomize sample & OCT ---
    samp = _Sample(sz_m, sy_m, sx_m); samp.randomize(N_scatterers)
    octsys = _OCT()

    # --- scan geometry & acquisition clock ---
    x_scan_range = _np.arange(x_scan_start, x_scan_stop, x_step)
    y_scan_range = _np.arange(y_scan_start, y_scan_stop, y_step)
    n_scans = len(x_scan_range)*len(y_scan_range)*n_images
    t_arr = _np.arange(0, n_scans*dt, dt)

    # --- motion traces, reference image is zero-motion ---
    zsig, ysig, xsig = motion_sigma_xyz
    dz_trace = _np.random.randn(len(t_arr))*zsig
    dy_trace = _np.random.randn(len(t_arr))*ysig
    dx_trace = _np.random.randn(len(t_arr))*xsig
    scans_per_vol = len(x_scan_range)*len(y_scan_range)
    dz_trace[:scans_per_vol] = 0.0
    dy_trace[:scans_per_vol] = 0.0
    dx_trace[:scans_per_vol] = 0.0

    # --- persist metadata ---
    os.makedirs(info_dir, exist_ok=True)
    _np.save(os.path.join(info_dir,'dz_trace.npy'), dz_trace)
    _np.save(os.path.join(info_dir,'dy_trace.npy'), dy_trace)
    _np.save(os.path.join(info_dir,'dx_trace.npy'), dx_trace)
    _np.save(os.path.join(info_dir,'x_scan_range.npy'), x_scan_range)
    _np.save(os.path.join(info_dir,'y_scan_range.npy'), y_scan_range)

    # --- simulate & save images/bscans with identical shapes/limits ---
    os.makedirs(output_root, exist_ok=True)
    t_index = 0
    figure_made = False
    fig = None; ax = None

    for v in range(n_images):
        bscan_folder = os.path.join(output_root, f'{v:05d}')
        bscan_png_folder = os.path.join(output_root, f'{v:05d}_png')
        os.makedirs(bscan_folder, exist_ok=True)
        if save_png: os.makedirs(bscan_png_folder, exist_ok=True)

        for bscan_index, y in enumerate(y_scan_range):
            bscan = []
            for x in x_scan_range:
                ss = octsys.spectral_scan(samp, y, x)
                ascan = _np.fft.fft(ss)
                bscan.append(ascan)
                # apply motion step
                samp.move(dz_trace[t_index], dy_trace[t_index], dx_trace[t_index])
                t_index += 1

            bscan = _np.array(bscan).T
            bscan = bscan[:octsys.N//2, :]  # keep one conjugate side
            _np.save(os.path.join(bscan_folder, f'bscan_{bscan_index:05d}.npy'), bscan)

            if save_png:
                if not figure_made:
                    bsy, bsx = bscan.shape
                    fig = _plt.figure(figsize=(bsx/50, bsy/50))
                    ax = fig.add_axes([0,0,1,1]); ax.set_xticks([]); ax.set_yticks([])
                    figure_made = True
                ax.clear()
                ax.imshow(_np.abs(bscan), cmap='gray', clim=(0,120))
                _plt.savefig(os.path.join(bscan_png_folder, f'bscan_{bscan_index:05d}.png'), dpi=100)

    return dict(
        output_root=output_root, info_dir=info_dir,
        x_scan_range=x_scan_range, y_scan_range=y_scan_range,
        n_images=n_images
    )

# --- registration of two synthetic images ---
@staticmethod
def register_synthetic_pair(
    data_root='bscans_synthetic',
    ref_index=0,
    target_index=1,
    sigma=5.0,
    compare_to_trace=True,
    info_folder='bscans_synthetic_info',
    samples_per_meter=5e5,
    show_plots=True
):
    """
    Recreates register_synthetic_images.py approach:
      - Loads images from disk
      - Row-wise Gaussian windowing; 3D phase correlation
      - Wrap-to-negative index handling (nxcswap)
      - Optional comparison to ground-truth motion traces
    Returns: (yshifts, zshifts, xshifts) as lists of ints (length = sy)
    """
    import numpy as _np
    from matplotlib import pyplot as _plt
    

    def _load_image(idx):
        folder = os.path.join(data_root, f'{idx:05d}')
        files = _glob.glob(os.path.join(folder, 'bscan*.npy'))
        files.sort()
        vol = [_np.load(f) for f in files]
        return _np.array(vol)  # (sy, sz, sx)

    ref = _load_image(ref_index)
    target = _load_image(target_index)

    sy, sz, sx = ref.shape
    y_arr = _np.arange(sy)

    def _nxcswap(a, N):  # identical behavior  :contentReference[oaicite:4]{index=4}
        return a - N if a > N//2 else a

    yshifts = []; zshifts = []; xshifts = []
    for y in range(sy):
        g = _np.exp(-(y_arr - y)**2 / (2*sigma**2))
        tar_t = _np.transpose(target, (1,2,0))
        tar_w = tar_t * g
        tar = _np.transpose(tar_w, (2,0,1))

        nxc3 = _np.abs(_np.fft.ifftn(_np.fft.fftn(tar) * _np.conj(_np.fft.fftn(ref))))
        yy, zz, xx = _np.unravel_index(_np.argmax(nxc3), nxc3.shape)

        yshifts.append(_nxcswap(yy, sy))
        zshifts.append(_nxcswap(zz, sz))
        xshifts.append(_nxcswap(xx, sx))

    if compare_to_trace:
        # Mirror the motion-trace comparison from the script  :contentReference[oaicite:5]{index=5}
        dx_trace = _np.load(os.path.join(info_folder,'dx_trace.npy'))
        dy_trace = _np.load(os.path.join(info_folder,'dy_trace.npy'))
        dz_trace = _np.load(os.path.join(info_folder,'dz_trace.npy'))

        n_ascans = sy * sx * 2  # two images used in original comparison
        n_bscans = sy * 2
        dx_trace = _np.cumsum(dx_trace[:n_ascans]) * samples_per_meter
        dy_trace = _np.cumsum(dy_trace[:n_ascans]) * samples_per_meter
        dz_trace = _np.cumsum(dz_trace[:n_ascans]) * samples_per_meter

        clock_ascan = _np.arange(n_ascans) * 10e-6
        clock_bscan = _np.arange(n_bscans) * 10e-6 * sx

        if show_plots:
            _plt.subplot(1,3,1)
            _plt.plot(clock_ascan, dx_trace, label='dx_trace')
            _plt.plot(clock_bscan[sy:], xshifts, label='x shift')  # align to 2nd image start
            _plt.legend()
            _plt.subplot(1,3,2)
            _plt.plot(clock_ascan, dy_trace, label='dy_trace')
            _plt.plot(clock_bscan[sy:], yshifts, label='y shift')
            _plt.legend()
            _plt.subplot(1,3,3)
            _plt.plot(clock_ascan, dz_trace, label='dz_trace')
            _plt.plot(clock_bscan[sy:], zshifts, label='z shift')
            _plt.legend()
            _plt.show()

    return yshifts, zshifts, xshifts

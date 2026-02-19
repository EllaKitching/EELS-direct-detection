"""
Enhanced Batch EELS / ADF / EDS processing pipeline
Summary of functionality:
 * Zero-loss peak alignment (LL to HL)
 * NNLS classification mapping with Ce3+/Ce4+ references
 * Distance-based masking for edge vs center spectra
 * EDS peak mapping (Ce, Pd)

Usage example:
python batch_eels_processing.py \
  --input_dir "C:\...\hspy" \
  --out_dir "C:\...\hspy_processed" \
  --ref_ce3 "path\to\ce3_ref.hspy" \
  --ref_ce4 "path\to\ce4_ref.hspy" \
  --preserve_subfolders \ # optional, to maintain input folder structure
  --nproc 4 \ # select based on your CPU cores to speed up analysis with multiprocessing

Other options:
-- skip_rl : skip RL deconvolution if no thickness effects are observed/ for quick test runs

Author: Ella Kitching, Cardiff University, 2025
License: GPL-3.0

  """

import warnings
warnings.filterwarnings("ignore", message=r"Pandas requires version '.*' or newer of 'numexpr'.*", category=UserWarning)

import os
os.environ['HYPERSPY_NO_UI_PROMPT'] = '1'  # Disable HyperSpy GUI
import matplotlib
matplotlib.use('Agg') 

import argparse
from multiprocessing import Pool, cpu_count
import json
import gc
import traceback

import numpy as np
import hyperspy.api as hs

# Stop HyperSpy GUI calls in batch mode
def _no_gui(*args, **kwargs):
    return None
import hyperspy.ui_registry
hyperspy.ui_registry.get_gui = _no_gui

from scipy import ndimage as ndi
from scipy.optimize import nnls
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import median_filter, gaussian_filter1d, uniform_filter1d, gaussian_filter

from skimage.measure import block_reduce

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib import cm

from tqdm import tqdm

# Functions
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def remove_spectral_spikes(spectrum, threshold_sigma=4.0, window=5):
    """
    Remove extreme positive and negative spikes from spectrum using median filtering.
    
    Parameters:
        spectrum : ndarray
            Input spectrum
        threshold_sigma : float
            Number of standard deviations to use as threshold (default: 4.0)
        window : int
            Window size for median filter (default: 5)
    
    Returns:
        cleaned : ndarray
            Spectrum with spikes removed
    """
    
    # Force computation if this is a lazy dask array
    if hasattr(spectrum, 'compute'):
        spectrum = spectrum.compute()
    
    cleaned = spectrum.copy()
    filtered = median_filter(cleaned, size=window, mode='reflect')
    residuals = cleaned - filtered
    residual_std = np.std(residuals)
    threshold = threshold_sigma * residual_std
    spike_mask = np.abs(residuals) > threshold
    cleaned[spike_mask] = filtered[spike_mask]
    
    num_spikes = np.sum(spike_mask)
    if num_spikes > 0:
        pass  # can add debug print if needed
    
    return cleaned


def normalise_spectra(spectrum):
    """
    Normalise spectrum to range [0, 1].
    
    Parameters:
        spectrum : ndarray
            Input spectrum
    
    Returns:
        ndarray
            Normalised spectrum
    """
    spec_min = float(np.min(spectrum))
    spec_max = float(np.max(spectrum))
    if spec_max - spec_min < 1e-12:
        return spectrum
    return (spectrum - spec_min) / (spec_max - spec_min)


def clean_and_normalise_spectrum(energy, spectrum, prominence=0.05, width=(1,5), pad=3, neg_prominence=None, clip_negative=True):
    """
    Detect narrow positive spikes and negative dips, interpolate across them, then normalise.
    
    Parameters:
        energy : 1D ndarray
            Energy axis (same length as spectrum)
        spectrum : 1D ndarray
            Input spectrum
        prominence : float, optional
            Prominence for scipy.signal.find_peaks (default: 0.05)
        width : tuple, optional
            Width range for peak detection (default: (1,5))
        pad : int, optional
            Padding around detected peaks for interpolation (default: 3)
        neg_prominence : float or None, optional
            Prominence for negative dips. If None, uses same as prominence (default: None)
        clip_negative : bool, optional
            If True, clips negative values to 0 after interpolation (default: True)
    
    Returns:
        tuple (norm_spec, clean_spec, peaks)
            norm_spec : 1D ndarray
                Normalised cleaned spectrum
            clean_spec : 1D ndarray
                Cleaned spectrum (not normalised)
            peaks : dict
                Dictionary with keys 'pos' and 'neg' containing peak indices
    """
    spec = np.array(spectrum, dtype=float)
    en = np.array(energy, dtype=float)

    # detect positive spikes
    pos_peaks, _ = find_peaks(spec, prominence=prominence, width=width)
    # detect negative spikes
    if neg_prominence is None:
        neg_prominence = prominence
    neg_peaks, _ = find_peaks(-spec, prominence=neg_prominence, width=width)

    # build mask excluding small regions around spikes/dips
    mask = np.ones_like(spec, dtype=bool)
    for p in np.concatenate([pos_peaks, neg_peaks]):
        lo = max(0, int(p - pad))
        hi = min(len(spec), int(p + pad + 1))
        mask[lo:hi] = False

    # require at least two good points for interpolation
    if np.sum(mask) < 2:
        # fallback: smooth small spikes with median filter-ish behaviour then normalise
        clean_spec = spec.copy()
    else:
        clean_spec = spec.copy()
        clean_spec[~mask] = np.interp(en[~mask], en[mask], spec[mask])

    # optionally remove small remaining negatives
    if clip_negative:
        clean_spec = np.clip(clean_spec, 0.0, None)

    # Normalise by integrated area (safe against zero)
    area = np.trapz(clean_spec, en)
    norm_spec = clean_spec / area if area != 0 else clean_spec

    peaks = {'pos': pos_peaks, 'neg': neg_peaks}
    return norm_spec, clean_spec, peaks


def remove_background_powerlaw(hs_signal, pre_edge_range=(100, 250)):
    """
    Per-pixel power-law background subtraction using HyperSpy's built-in method.
    
    Parameters:
        hs_signal : hyperspy.signal.Signal1D
            Input EELS signal
        pre_edge_range : tuple, optional
            Energy values (eV) for background fitting window (default: (100, 250))
    
    Returns:
        hyperspy.signal.Signal1D
            Signal with background removed
    """
    try:
        original_shape = hs_signal.data.shape
        original_offset = hs_signal.axes_manager[-1].offset
        original_scale = hs_signal.axes_manager[-1].scale
        original_units = hs_signal.axes_manager[-1].units
        original_size = hs_signal.axes_manager[-1].size

        s_bg = hs_signal.deepcopy()

        energy_axis = s_bg.axes_manager[-1].axis
        if pre_edge_range[0] < energy_axis.min():
            pre0_ev = float(energy_axis[int(pre_edge_range[0])])
            pre1_ev = float(energy_axis[min(int(pre_edge_range[1]), len(energy_axis)-1)])
        else:
            pre0_ev = float(pre_edge_range[0])
            pre1_ev = float(pre_edge_range[1])

        # Suppress RuntimeWarnings from Hyperspy power-law fits (log/divide issues on zeros, this doesnt effect results significantly)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in log')
            warnings.filterwarnings('ignore', 'divide by zero encountered')
            warnings.filterwarnings('ignore', 'overflow encountered')
            s_bg.remove_background(signal_range=(pre0_ev, pre1_ev), background_type='PowerLaw')

        return s_bg
    except Exception as e:
        #  background removal failure, proceed without
        return hs_signal

def save_figure(fig, outpath, dpi=200, tight=True, pad_inches=0.02):
    """
    Save a matplotlib figure with optional tight bounding box cropping.

    Parameters:
        fig : matplotlib.figure.Figure
            Figure to save
        outpath : str
            Output filename
        dpi : int, optional
            Resolution (default: 200)
        tight : bool, optional
            When True use bbox_inches='tight' to trim whitespace. For image
            maps (e.g., MLLS/NNLS), set to False to avoid edge clipping (default: True)
        pad_inches : float, optional
            Padding used when tight=True (default: 0.02)
    
    Returns:
        None, but saves figure to outpath
    """
    try:
        if tight:
            fig.savefig(outpath, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
        else:
            # No tight cropping – preserves full image extent
            fig.savefig(outpath, dpi=dpi)
    except ValueError as e:
        if "Image size" in str(e) and "too large" in str(e):
            # bbox_inches='tight' sometimes miscalculates, retry without it
            fig.savefig(outpath, dpi=dpi)
        else:
            raise
    finally:
        plt.close(fig)


def add_scalebar(ax, pixel_size_nm=0.15, scale_length_nm=5, location='lower left', 
                 color='white', fontsize=12, linewidth=3, outlinewidth=3):
    """
    Add a scale bar to an image axis consistently.
    Drawn as a filled rectangle with a black edge so all four sides are outlined.

    Parameters:
        ax : matplotlib.axes.Axes
            Axis to draw the scale bar on
        pixel_size_nm : float, optional
            Size of one pixel in nanometres (default: 0.15)
        scale_length_nm : float, optional
            Desired length of the scale bar in nanometres (default: 5)
        location : str, optional
            Location of the scale bar, e.g. 'lower left', 'lower right' (default: 'lower left')
        color : str, optional
            Colour of the scale bar and label text (default: 'white')
        fontsize : int, optional
            Font size for the label in points (default: 12)
        linewidth : float, optional
            Thickness of the scale bar in points (default: 3)
        outlinewidth : float, optional
            Width of the black outline stroke in points (default: 3)
    
    Returns:
        None, but adds scale bar to the provided axis
    """

    scale_pixels = scale_length_nm / pixel_size_nm

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    img_width = xlim[1] - xlim[0]
    img_height = ylim[0] - ylim[1]

    if 'left' in location.lower():
        x_offset = img_width * 0.03
    else:
        x_offset = img_width * 0.97 - scale_pixels

    if 'lower' in location.lower():
        y_center = img_height * 0.95
    else:
        y_center = img_height * 0.05

    # Convert linewidth in points to data units (pixels) using figure DPI
    dpi = ax.figure.dpi
    bar_thickness_px = max(1.0, (linewidth) * (dpi / 72.0))  # 1 pt = 1/72 inch

    # Draw scale bar with black outline on all sides
    rect = Rectangle(
        (x_offset, y_center - bar_thickness_px / 2.0),
        width=scale_pixels,
        height=bar_thickness_px,
        facecolor=color,
        edgecolor='black',
        linewidth=max(1.0, outlinewidth * 0.6),
        joinstyle='miter'
    )
    ax.add_patch(rect)

    # Label
    text_x = x_offset + scale_pixels / 2.0
    text_y = y_center - img_height * 0.015
    text = ax.text(text_x, text_y, f'{scale_length_nm:.0f} nm',
                   color=color, fontsize=fontsize, ha='center', va='bottom',
                   fontweight='bold')
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='black'),
        path_effects.Normal()
    ])

def imshow_with_black(ax, data, cmap='viridis', vmin=None, vmax=None, **kwargs):
    """
    Display data (can be a masked array) with masked values rendered as black.
    
    Parameters:
        ax : matplotlib.axes.Axes
            Axis to draw on
        data : ndarray or masked array
            Image data to display
        cmap : str or colormap, optional
            Colormap to use (default: 'viridis')
        vmin : float or None, optional
            Minimum value for colormap (default: None)
        vmax : float or None, optional
            Maximum value for colormap (default: None)
        **kwargs : dict
            Additional keyword arguments passed to imshow
    
    Returns:
        matplotlib.image.AxesImage
            The image object
    """
    cmap_obj = mpl.colormaps.get(cmap) if isinstance(cmap, str) else cmap
    samples = cmap_obj(np.linspace(0, 1, 256))

    cmap_copy = ListedColormap(samples)
    cmap_copy.set_bad('black')
    # Set interpolation='none' to avoid thick borders at mask edges
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'none'
    im = ax.imshow(data, cmap=cmap_copy, vmin=vmin, vmax=vmax, **kwargs)
    return im

def validate_peak_width(signal_1d, peak_idx, energy_axis, min_width_ev=2.0):
    """
    Validate that a peak has at least a minimum physical width in eV.

    Parameters:
        signal_1d : 1D ndarray
            Signal with non-negative values
        peak_idx : int
            Index of the peak within signal_1d 
        energy_axis : 1D ndarray
            Energy axis of same length as signal_1d (eV)
        min_width_ev : float, optional
            Minimum acceptable width at half-prominence in eV (default: 2.0)

    Returns:
        tuple (is_valid, width_ev)
            is_valid : bool
                True if peak width meets minimum requirement
            width_ev : float
                Measured peak width in eV
    """
    try:
        if signal_1d is None or energy_axis is None:
            return False, 0.0
        n = len(signal_1d)
        if n == 0 or peak_idx < 0 or peak_idx >= n:
            return False, 0.0
        # Compute peak widths at half prominence using scipy.signal.peak_widths
        widths, width_heights, left_ips, right_ips = peak_widths(signal_1d, [peak_idx], rel_height=0.5)
        # Convert sample width to eV using local linear interpolation of energy axis
        left_e = np.interp(left_ips[0], np.arange(n), energy_axis)
        right_e = np.interp(right_ips[0], np.arange(n), energy_axis)
        width_ev = float(max(0.0, right_e - left_e))
        return (width_ev >= float(min_width_ev), width_ev)
    except Exception:
        return False, 0.0, print("Warning: Peak width validation failed.")


def nnls_worker(args):
    """
    Worker function for NNLS/MLLS pixel fitting with optional background masking.
    
    Parameters:
        args : tuple
            (i, j, pixel_spectrum, A, bg_threshold, is_background)
            i, j : int
                Pixel coordinates
            pixel_spectrum : 1D ndarray
                Spectrum at pixel (i, j)
            A : 2D ndarray
                Reference matrix for fitting
            bg_threshold : float or None
                Background threshold
            is_background : bool
                Whether pixel is background
    
    Returns:
        tuple (i, j, nnls_c3, nnls_c4, mlls_c3, mlls_c4)
            Pixel coordinates and fitted coefficients
    """
    i, j, pixel_spectrum, A, bg_threshold, is_background = args
    try:
        if is_background:
            return (i, j, 0.0, 0.0, 0.0, 0.0)
        
        try:
            pixel_spectrum_norm = normalise_spectra(pixel_spectrum)
            coeffs_nnls, _ = nnls(A, pixel_spectrum_norm)
        except Exception:
            coeffs_nnls = np.array([0.0, 0.0])

        total_nnls = coeffs_nnls.sum()
        if total_nnls < 1e-12:
            nnls_c3 = nnls_c4 = 0.0
        else:
            nnls_c3 = float(coeffs_nnls[0] / total_nnls)
            nnls_c4 = float(coeffs_nnls[1] / total_nnls)

        try:
            coeffs_mlls, *_ = np.linalg.lstsq(A, pixel_spectrum, rcond=None)
            coeffs_mlls = np.asarray(coeffs_mlls).flatten()
        except Exception:
            coeffs_mlls = np.array([0.0, 0.0])

        try:
            coeffs_mlls = np.clip(coeffs_mlls, 0.0, None)
        except Exception:
            pass

        total_mlls = float(coeffs_mlls.sum())
        if total_mlls < 1e-12:
            mlls_c3 = float(nnls_c3)
            mlls_c4 = float(nnls_c4)
        else:
            mlls_c3 = float(coeffs_mlls[0] / total_mlls)
            mlls_c4 = float(coeffs_mlls[1] / total_mlls)

        return (i, j, nnls_c3, nnls_c4, mlls_c3, mlls_c4)
    except Exception:
        return (i, j, 0.0, 0.0, 0.0, 0.0)


def create_eds_peak_maps(eds_signal, elements=['Ce', 'Pd']):
    """
    Create elemental maps from EDS signal.
    
    Parameters:
        eds_signal : hyperspy EDS signal
            EDS data signal
        elements : list, optional
            List of element symbols to map (default: ['Ce', 'Pd'])
    
    Returns:
        dict
            Dictionary with element line keys and map arrays as values
    """
    maps = {}
    for element in elements:
        if element == 'Ce':
            lines = ['Ce_La', 'Ce_Ma']
        elif element == 'Pd':
            lines = ['Pd_La', 'Pd_Ka']
        
        for line in lines:
            element_map = eds_signal.get_lines_intensity([line])[0]
            maps[f'{element}_{line}'] = element_map.data

    return maps

def plot_eds_maps_overlay(eds_maps, out_dir, base_name, pixel_size_nm=0.15):
    """
    Create EDS visualization with black background (Ce=yellow, Pd=pink).

    Parameters:
        eds_maps : dict
            Dictionary of elemental maps from create_eds_peak_maps
        out_dir : str
            Output directory
        base_name : str
            Base filename
        pixel_size_nm : float, optional
            Pixel size in nanometres (default: 0.15)
    
    Returns:
        str or None
            Path to saved figure, or None if failed
    
    Notes:
        - Left: Ce map (YlOrBr colormap)
        - Middle: Pd map (RdPu colormap)
        - Right: RGB overlay with Ce→(R+G)=yellow, Pd→(R+B)=pink
    """
    try:
        figures_dir = os.path.join(out_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        ce_map = None
        pd_map = None
        for key, val in eds_maps.items():
            if ce_map is None and key.startswith('Ce_'):
                ce_map = np.asarray(val)
            if pd_map is None and key.startswith('Pd_'):
                pd_map = np.asarray(val)
            if ce_map is not None and pd_map is not None:
                break

        # remove NaNs/Infs and apply very light smoothing for display
        if ce_map is not None:
            ce_map = np.nan_to_num(ce_map, nan=0.0, posinf=0.0, neginf=0.0)
            ce_map = gaussian_filter(ce_map, sigma=1.5)
        if pd_map is not None:
            pd_map = np.nan_to_num(pd_map, nan=0.0, posinf=0.0, neginf=0.0)
            pd_map = gaussian_filter(pd_map, sigma=1.5)

        fig = plt.figure(figsize=(10, 4), facecolor='black')
        gs = plt.GridSpec(1, 4, figure=fig, wspace=0.1, hspace=0.1)
        ax1 = fig.add_subplot(gs[0, 0], facecolor='black')
        ax2 = fig.add_subplot(gs[0, 1], facecolor='black')
        ax3 = fig.add_subplot(gs[0, 2:], facecolor='black')

        if ce_map is not None:
            from matplotlib.colors import LinearSegmentedColormap
            colors_ce = [(0, 0, 0), (1, 1, 0)]  # black to yellow
            ce_cmap = LinearSegmentedColormap.from_list('ce_yellow', colors_ce, N=256)
            ce_cmap.set_under('black')
            ce_cmap.set_bad('black')
            
            # Create masked array - mask out ONLY truly zero/background pixels
            ce_threshold = max(np.percentile(ce_map[ce_map > 0], 0.5), 1e-9) if np.any(ce_map > 0) else 1e-9
            ce_masked = np.ma.masked_where(ce_map <= ce_threshold, ce_map)
            vmax_ce = float(np.nanmax(ce_map)) if np.nanmax(ce_map) > 0 else 1.0
            
            im1 = ax1.imshow(ce_masked, cmap=ce_cmap, vmin=ce_threshold, vmax=vmax_ce, 
                           interpolation='none', aspect='equal', origin='upper')
            ax1.set_title('Ce', color='white', fontsize=12)
            ax1.set_xlim(-0.5, ce_map.shape[1] - 0.5)
            ax1.set_ylim(ce_map.shape[0] - 0.5, -0.5)
            add_scalebar(ax1, pixel_size_nm=pixel_size_nm, scale_length_nm=5, 
                         location='lower left', color='white', fontsize=10, linewidth=2)
            ax1.axis('off')

        if pd_map is not None:
            from matplotlib.colors import LinearSegmentedColormap
            colors_pd = [(0, 0, 0), (1, 0, 1)]  # black to magenta 
            pd_cmap = LinearSegmentedColormap.from_list('pd_pink', colors_pd, N=256)
            pd_cmap.set_under('black')
            pd_cmap.set_bad('black')
            
            # Create masked array - mask out ONLY truly zero/background pixels
            pd_threshold = max(np.percentile(pd_map[pd_map > 0], 0.5), 1e-9) if np.any(pd_map > 0) else 1e-9
            pd_masked = np.ma.masked_where(pd_map <= pd_threshold, pd_map)
            vmax_pd = float(np.nanmax(pd_map)) if np.nanmax(pd_map) > 0 else 1.0
            
            im2 = ax2.imshow(pd_masked, cmap=pd_cmap, vmin=pd_threshold, vmax=vmax_pd, 
                           interpolation='none', aspect='equal', origin='upper')
            ax2.set_title('Pd', color='white', fontsize=12)
            ax2.set_xlim(-0.5, pd_map.shape[1] - 0.5)
            ax2.set_ylim(pd_map.shape[0] - 0.5, -0.5)
            add_scalebar(ax2, pixel_size_nm=pixel_size_nm, scale_length_nm=5, 
                         location='lower left', color='white', fontsize=10, linewidth=2)
            ax2.axis('off')

    # Combined overlay on black
        if ce_map is not None or pd_map is not None:
            # Determine shape from available map
            shape = ce_map.shape if ce_map is not None else pd_map.shape
            overlay = np.zeros((shape[0], shape[1], 3), dtype=float)

            if ce_map is not None:
                # Threshold to avoid amplifying noise in background
                ce_threshold = max(np.percentile(ce_map[ce_map > 0], 0.5), 1e-9) if np.any(ce_map > 0) else 1e-9
                ce_signal = np.where(ce_map >= ce_threshold, ce_map, 0.0)
                ce_max = float(np.max(ce_signal)) if np.max(ce_signal) > 0 else 1.0
                ce_norm = ce_signal / ce_max
                overlay[:, :, 0] += ce_norm  # R
                overlay[:, :, 1] += ce_norm  # yellow

            if pd_map is not None:
                # Threshold to avoid amplifying noise in background
                pd_threshold = max(np.percentile(pd_map[pd_map > 0], 0.5), 1e-9) if np.any(pd_map > 0) else 1e-9
                pd_signal = np.where(pd_map >= pd_threshold, pd_map, 0.0)
                pd_max = float(np.max(pd_signal)) if np.max(pd_signal) > 0 else 1.0
                pd_norm = pd_signal / pd_max
                overlay[:, :, 0] += pd_norm  # add to R (mix)
                overlay[:, :, 2] += pd_norm  # pink

            # Clip to [0,1] to avoid overflow
            overlay = np.clip(overlay, 0.0, 1.0)
            ax3.imshow(overlay, interpolation='none', aspect='equal', origin='upper')
            ax3.set_title('Ce (yellow) + Pd (pink) Overlay', color='white', fontsize=12)
            ax3.set_xlim(-0.5, shape[1] - 0.5)
            ax3.set_ylim(shape[0] - 0.5, -0.5)
            add_scalebar(ax3, pixel_size_nm=pixel_size_nm, scale_length_nm=5, 
                         location='lower left', color='white', fontsize=10, linewidth=2)
            ax3.axis('off')

        try:
            plt.tight_layout()
        except Exception:
            pass  # Ignore tight_layout warnings/errors
        fig_path = os.path.join(figures_dir, f"{base_name}_eds_maps_overlay.png")
        # Save without tight cropping (retain original visual style)
        save_figure(fig, fig_path, tight=False)
        print(f"  Saved EDS overlay: {fig_path}")
        return fig_path
    except Exception as e:
        print(f"Warning: EDS overlay plotting failed: {e}")
        return None


class DatasetLoader:
    """
    Handles loading and validation of complete EELS/ADF/EDS datasets.
    
    Attributes:
        base_path : str
            Base path for dataset
        signals : dict
            Dictionary of loaded signals
        is_valid : bool
            Whether dataset loaded successfully
        errors : list
            List of error messages
    """
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.base_name = None
        self.signals = {}
        self.is_valid = False
        self.metadata = {}
        self.errors = []
    
    def find_companion_files(self, summed_path):
        """Find all related files for a dataset"""
        try:
            if not os.path.exists(summed_path):
                raise FileNotFoundError(f"Source file not found: {summed_path}")
                
            folder = os.path.dirname(summed_path)
            base_name = os.path.splitext(os.path.basename(summed_path))[0]
            # Extract the base identifier (e.g., "18pA_InSitu_(3)" from "18pA_InSitu_(3)_HL_stack_sumall")
            base = base_name
            import re
            
            # match and remove patterns like _HL_stack_sum*, _LL_stack_sum*, etc.
            pattern = r'_(HL|LL|ADF|EDS)_stack(_sum(all_)?\d+|_summed)?$'
            match = re.search(pattern, base, re.IGNORECASE)
            if match:
                base = base[:match.start()]
            else:
                # Try generic removal if regex didn't match
                for suffix in ['_summed', '_stack', '_hl', '_ll', '_eds', '_adf']:
                    if base.lower().endswith(suffix):
                        base = base[:base.lower().rfind(suffix)]
                        break
            
            print(f"\nSearching for files matching: {base}")
            
            files = {
                'HL_summed': {'path': None, 'required': True, 'found': False},
                'LL_summed': {'path': None, 'required': False, 'found': False},
                'ADF': {'path': None, 'required': False, 'found': False},
                'EDS': {'path': None, 'required': False, 'found': False}
            }
            
            all_files = os.listdir(folder)
            matching_files = [f for f in all_files if base in f and f.lower().endswith('.hspy')]
            
            for f in matching_files:
                f_lower = f.lower()
                full_path = os.path.join(folder, f)
                
                try:
                    if '_hl' in f_lower and ('_sumall' in f_lower or '_sum5' in f_lower or '_sum10' in f_lower or '_sum20' in f_lower):
                        files['HL_summed']['path'] = full_path
                        files['HL_summed']['found'] = True
                    elif '_ll' in f_lower and ('_sumall' in f_lower or '_sum5' in f_lower or '_sum10' in f_lower or '_sum20' in f_lower):
                        files['LL_summed']['path'] = full_path
                        files['LL_summed']['found'] = True
                    elif '_adf' in f_lower or 'haadf' in f_lower or 'hadf' in f_lower:
                        files['ADF']['path'] = full_path
                        files['ADF']['found'] = True
                    elif '_eds' in f_lower:
                        files['EDS']['path'] = full_path
                        files['EDS']['found'] = True
                except Exception as e:
                    self.errors.append(f"Error processing file {f}: {str(e)}")
            
            for signal_type, info in files.items():
                if info['found']:
                    print(f"  Found: {signal_type}")
            
            return files
            
        except Exception as e:
            self.errors.append(f"Error in find_companion_files: {str(e)}")
            return None
    
    def validate_dataset(self, files):
        """Check if all required files are present and validate formats"""
        if not files:
            return False
            
        validation_results = {
            'missing_required': [],
            'missing_optional': [],
            'invalid_files': []
        }
        
        for signal_type, info in files.items():
            if not info['found']:
                if info['required']:
                    validation_results['missing_required'].append(signal_type)
                else:
                    validation_results['missing_optional'].append(signal_type)
                continue
                
            try:
                if info['path'] and os.path.exists(info['path']):
                    with open(info['path'], 'rb') as f:
                        if not any(x in f.read(8).decode(errors='ignore') for x in ['HDF', 'HSpy']):
                            validation_results['invalid_files'].append(signal_type)
            except Exception as e:
                self.errors.append(f"Error validating {signal_type}: {str(e)}")
                validation_results['invalid_files'].append(signal_type)
        
        self.metadata['validation'] = validation_results
        
        if validation_results['missing_required'] or validation_results['invalid_files']:
            error_msg = []
            if validation_results['missing_required']:
                error_msg.append(f"Missing required files: {validation_results['missing_required']}")
            if validation_results['invalid_files']:
                error_msg.append(f"Invalid files: {validation_results['invalid_files']}")
            if validation_results['missing_optional']:
                print(f"Warning: Missing optional files: {validation_results['missing_optional']}")
            
            if error_msg:
                self.errors.append(" | ".join(error_msg))
            return False
        return True
    
    def load_dataset(self, summed_path):
        """Load and validate a complete dataset"""
        self.signals.clear()
        self.errors = []
        
        try:
            files = self.find_companion_files(summed_path)
            if not self.validate_dataset(files):
                return False
            
            total_files = sum(1 for info in files.values() if info['path'])
            
            for signal_type, info in files.items():
                if not info['path']:
                    continue
                    
                try:
                    self.signals[signal_type] = hs.load(info['path'])
                except Exception as e:
                    self.errors.append(f"Error loading {signal_type}: {str(e)}")
                    if info['required']:
                        return False
            
            if not self._verify_signal_compatibility():
                return False
            
            self.is_valid = True
            print(f"Loaded signals: {list(self.signals.keys())}")
            return True
            
        except Exception as e:
            self.errors.append(f"Critical error in load_dataset: {str(e)}")
            self.signals.clear()
            self.is_valid = False
            return False
    
    def _verify_signal_compatibility(self):
        """Verify that loaded signals are compatible"""
        try:
            if 'HL_summed' not in self.signals:
                self.errors.append("Missing required HL_summed signal")
                return False
            
            hl_signal = self.signals['HL_summed']
            if not hasattr(hl_signal, 'metadata') or not hasattr(hl_signal.metadata, 'Signal'):
                self.errors.append("HL_summed missing metadata")
                return False
                
            if hl_signal.metadata.Signal.signal_type != "EELS":
                self.errors.append(f"HL_summed wrong signal type: {hl_signal.metadata.Signal.signal_type}")
                return False
                
            base_shape = hl_signal.data.shape
            
            for sig_type in ['LL_summed', 'ADF']:
                if sig_type in self.signals:
                    sig = self.signals[sig_type]
                    if not hasattr(sig, 'data'):
                        self.errors.append(f"{sig_type} has no data attribute")
                        continue
                        
                    sig_shape = sig.data.shape
                    if sig_shape[:2] != base_shape[:2]:
                        self.errors.append(f"Shape mismatch in {sig_type}: {sig_shape} vs {base_shape}")
                        continue
            
            self.metadata['signal_info'] = {
                'HL_summed_shape': base_shape,
                'signal_type': hl_signal.metadata.Signal.signal_type,
                'available_signals': list(self.signals.keys())
            }
            
            return True
            
        except Exception as e:
            self.errors.append(f"Error in signal compatibility check: {str(e)}")
            return False
    
    def get_errors(self):
        """Return list of errors encountered during loading"""
        return self.errors

# Main processing function
def process_dataset_v3(summed_path, out_dir, refs, params, nproc=None):
    """
    Process a single EELS dataset with enhanced error handling.
    
    Parameters:
        summed_path : str
            Path to summed HL EELS file
        out_dir : str
            Output directory
        refs : dict
            Dictionary with 'ce3' and 'ce4' reference spectra
        params : dict
            Processing parameters
        nproc : int or None, optional
            Number of processes for multiprocessing (default: None, uses CPU count-1)
    
    Returns:
        bool
            True if processing succeeded, False otherwise
    """
    try:
        saved_figures = []
        base_name = os.path.splitext(os.path.basename(summed_path))[0]
        
        figures_dir = os.path.join(out_dir, 'figures')
        for d in [out_dir, figures_dir]:
            ensure_dir(d)
        
        # Check if already processed by looking for key output files
        postrl_npz = os.path.join(out_dir, f"{base_name}_postRL.npz")
        ce3_nnls = os.path.join(out_dir, f"{base_name}_bin2_ce3_map_nnls.npy")
        ce3_mlls = os.path.join(out_dir, f"{base_name}_bin2_ce3_map_mlls.npy")
        if os.path.exists(postrl_npz) and os.path.exists(ce3_nnls) and os.path.exists(ce3_mlls):
            print(f"\nSkipping {base_name} - already processed (output files exist)")
            return True
        
        # Discover files but DO NOT load everything into memory.
        # lazily to reduce peak memory usage
        loader = DatasetLoader(summed_path)
        files = loader.find_companion_files(summed_path)
        if not files or not files.get('HL_summed'):
            print(f"Failed to find HL stack for: {summed_path}")
            return False

        def _extract_path_multi(dct, keys):
            """Return the first non-empty path for any of the keys."""
            for k in keys:
                entry = dct.get(k)
                if entry is None:
                    continue
                if isinstance(entry, dict):
                    p = entry.get('path')
                else:
                    p = entry
                if p:
                    return p
            return None

        hl_path = _extract_path_multi(files, ['HL', 'HL_summed'])
        ll_path = _extract_path_multi(files, ['LL', 'LL_summed'])
        adf_path = _extract_path_multi(files, ['ADF'])
        eds_path = _extract_path_multi(files, ['EDS'])
        try:
            s_HL = hs.load(hl_path)
        except Exception as e:
            print(f"Error loading HL stack: {e}")
            traceback.print_exc()
            return False
        # placeholder for LL/ADF/EDS signals, will be loaded lazily when needed
        s_LL = None
        adf_sig = None
        eds_sig = None
        
        print(f"\nProcessing: {base_name}")
                
        adf_unbinned_data = None
        if adf_path is not None:
            try:
                adf_sig_tmp = hs.load(adf_path)
                if len(adf_sig_tmp.data.shape) == 3:
                    if adf_sig_tmp.data.shape[0] < adf_sig_tmp.data.shape[1] and adf_sig_tmp.data.shape[0] < adf_sig_tmp.data.shape[2]:
                        adf_data_summed = np.sum(adf_sig_tmp.data, axis=0)
                        adf_unbinned_data = adf_data_summed.copy()
                    else:
                        adf_unbinned_data = adf_sig_tmp.data.squeeze().copy()
                else:
                    adf_unbinned_data = adf_sig_tmp.data.squeeze().copy()
                del adf_sig_tmp
            except Exception:
                # If ADF fails to load, continue without it
                adf_unbinned_data = None
        
        # Match spatial dimensions
        spatial_shapes = []
        if len(s_HL.data.shape) >= 2:
            spatial_shapes.append(s_HL.data.shape[:2])
        if ll_path is not None:
            try:
                sig_ll = hs.load(ll_path)
                spatial_shapes.append(sig_ll.data.shape[:2])
                # Keep s_LL loaded for later use (alignment, RL deconvolution)
                s_LL = sig_ll
            except Exception:
                pass
        if adf_unbinned_data is not None:
            adf_shape = adf_unbinned_data.shape
            if len(adf_shape) >= 2:
                spatial_shapes.append(adf_shape[:2])
        if eds_path is not None:
            try:
                sig_eds = hs.load(eds_path)
                spatial_shapes.append(sig_eds.data.shape[:2])
                del sig_eds
            except Exception:
                pass
        
        if spatial_shapes:
            min_y = min(shape[0] for shape in spatial_shapes)
            min_x = min(shape[1] for shape in spatial_shapes)
            
            all_same = all(shape == (min_y, min_x) for shape in spatial_shapes)
            
            if not all_same:
                print(f"Matching to common size: ({min_y}, {min_x})")
                
                # Rebin/crop signals
                if adf_sig is not None:
                    adf_data_temp = adf_sig.data.squeeze()
                    current_shape = adf_data_temp.shape[:2]
                    if current_shape != (min_y, min_x):
                        if current_shape[0] == 2 * min_y and current_shape[1] == 2 * min_x:
                            adf_binned_data = block_reduce(adf_sig.data, block_size=(2, 2), func=np.mean)
                            adf_sig = hs.signals.Signal2D(adf_binned_data)
                        else:
                            adf_cropped_data = adf_sig.data[:min_y, :min_x]
                            adf_sig = hs.signals.Signal2D(adf_cropped_data)
                
                if s_HL.data.shape[:2] != (min_y, min_x):
                    s_HL = s_HL.inav[:min_y, :min_x]
                if s_LL is not None and s_LL.data.shape[:2] != (min_y, min_x):
                    s_LL = s_LL.inav[:min_y, :min_x]
                if eds_sig is not None and eds_sig.data.shape[:2] != (min_y, min_x):
                    eds_sig = eds_sig.inav[:min_y, :min_x]
        
        stack_variants = {'': s_HL}
        base_name_orig = base_name

        # Loop over variants and run the full processing pipeline for each
        for var_suffix, s_HL in stack_variants.items():
            if var_suffix:
                base_name = base_name_orig + var_suffix
            else:
                base_name = base_name_orig

            print ('HL data shape:',s_HL.data.shape, 'LL data shape:', s_LL.data.shape if s_LL is not None else None,)
            print ('HL data min:', s_HL.axes_manager[2].axis.min(), 'HL data max:', s_HL.axes_manager[2].axis.max())

            # Zero-loss alignment
            # Load LL (PSF) if not already loaded
            if s_LL is None and ll_path is not None:
                try:
                    s_LL = hs.load(ll_path)
                except Exception as e:
                    print(f"  Could not load LL (PSF) for alignment/deconvolution: {e}")
                    s_LL = None

            if s_LL is not None:
                try:
                    print("Aligning zero-loss peak")

                    # Apply offset corrections if specified
                    if 'eels_offset' in params and params['eels_offset'] is not None:
                        hl_offset_override = params['eels_offset']
                        original_hl_offset = s_HL.axes_manager[-1].offset
                        if abs(original_hl_offset - hl_offset_override) > 0.1:
                            s_HL.axes_manager[-1].offset = hl_offset_override
                            print('Offset override applied to HL:', hl_offset_override)
                            try:
                                hl_axis_after = s_HL.axes_manager[2].axis
                                print(f"  HL energy axis after override: {hl_axis_after.min():.1f} to {hl_axis_after.max():.1f} eV")
                            except Exception:
                                pass

                    # Apply LL offset correction if given
                    if params.get('ll_offset_correction') is not None:
                        ll_offset_correct = params.get('ll_offset_correction', -36.0)
                        s_LL.axes_manager[-1].offset = ll_offset_correct

                    # Perform ZLP alignment
                    try:
                        s_LL.align_zero_loss_peak(subpixel=True, also_align=[s_HL], crop=False)
                    except Exception:
                        print('Zero-loss alignment per-variant skipped or failed')

                except Exception as e:
                    print(f"Warning: Zero-loss alignment failed: {e}")

            # EDS overlay
            try:
                if eds_sig is not None:
                    eds_maps = create_eds_peak_maps(eds_sig)
                    plot_eds_maps_overlay(eds_maps, out_dir, base_name, pixel_size_nm=0.15)
            except Exception as e:
                print(f"Warning: EDS overlay skipped: {e}")

            eels_slice_idx = params.get('eels_slice_idx', [3102, 3417])
            
            target_spectral_size = refs['ce3'].data.shape[0]
            print(f"Reference spectrum size: {target_spectral_size}")

            processing_versions = []

            has_energy_axis = any('energy' in str(axis.name).lower() or 'eV' in str(axis.units) 
                                for axis in s_HL.axes_manager._axes)
            
            if len(s_HL.data.shape) == 3:
                if has_energy_axis:
                    s = s_HL 
                else:
                    raise ValueError(f"Dataset missing energy dimension")
                
            elif len(s_HL.data.shape) == 4:
                frame_axis = None
                for i, axis in enumerate(s_HL.axes_manager._axes):
                    if 'frame' in str(axis.name).lower() or axis.size < 50:
                        frame_axis = i
                        break
                
                if frame_axis is not None:
                    s = s_HL.mean(axis=frame_axis)
                else:
                    s = s_HL.mean(axis=2)
            else:
                s = s_HL


            # Check if the third dimension is reasonable for energy, as 3000+ channels is expected. Even with binning shouldnt be too small.
            if s.data.shape[2] < 100:  
                raise ValueError(f"Energy dimension too small: {s.data.shape[2]}")
                        
            # Background removal
            print("Removing background (power-law)")
            pre_edge_win = params.get('pre_edge_range', (700, 850))
            try:
                s_bg = remove_background_powerlaw(s, pre_edge_range=pre_edge_win)
            except Exception as e:
                print(f"Background removal failed: {e}")
                s_bg = s

            # Save pre-RL version for comparison
            s_bg_noRL = s_bg.deepcopy()
            
            # Richardson-Lucy deconvolution (optional)
            try:
                if params.get('skip_rl'):
                    print("Skipping RL deconvolution (skip_rl flag)")
                    s_bg_noRL = None
                elif s_LL is not None:
                    print("Applying Richardson-Lucy deconvolution using LL as PSF (10 iterations)")
                    workers = min(5, (nproc or max(1, cpu_count()-1)))
                    try:
                        s_bg = s_bg.richardson_lucy_deconvolution(psf=s_LL, iterations=10, num_workers=workers)
                    except Exception: # if issue with multiprocessing
                        s_bg = s_bg.richardson_lucy_deconvolution(psf=s_LL, iterations=10)
                    
                    # Clip negative artifacts from RL deconvolution
                    s_bg.data = np.clip(s_bg.data, 0.0, None)
                    print("  Clipped negative RL artifacts")
                else:
                    print("LL (PSF) not available: skipping RL deconvolution")
                    s_bg_noRL = None  
            except Exception as e:
                print(f"Warning: RL deconvolution failed or not available: {e}")
                s_bg_noRL = None  
            try:
                postrl_npz = os.path.join(out_dir, f"{base_name}_postRL.npz")
                try:
                    energy_axis_rl = np.asarray(s_bg.axes_manager[2].axis)
                except Exception:
                    energy_axis_rl = np.arange(s_bg.data.shape[2])

                # Save numerical arrays (data and energy). Hyperspy object can be reloaded separately if needed.
                np.savez(postrl_npz, data=np.array(s_bg.data, dtype=float), energy=energy_axis_rl)
                print(f"Saved post-RL data: {postrl_npz}")
            except Exception as e:
                print(f"Warning: could not save post-RL data: {e}")

            # Auto-calculate energy shift from reference using CLEANED data (after BG removal & RL deconvolution)
            energy_shift = params.get('energy_shift_correction', 0.0)
            if abs(energy_shift) < 0.01:
                print("\nFinding experimental M5 peak position to align references...")
                print("  (Using background-removed & deconvolved spectrum for better peak detection)")

                try:
                    summed_signal = s_bg.sum(axis=(0, 1))
                    summed_spectrum = summed_signal.data
                    
                    print(f"  Summed spectrum shape: {summed_spectrum.shape}")
                    print(f"  Summed spectrum range: {np.min(summed_spectrum):.1f} to {np.max(summed_spectrum):.1f}")

                    summed_spectrum_clean = remove_spectral_spikes(summed_spectrum, threshold_sigma=4.0, window=5)
                    summed_spectrum_norm = normalise_spectra(summed_spectrum_clean)

                    exp_energy_axis = s_bg.axes_manager[-1].axis
                    print(f"  Energy axis range: {exp_energy_axis.min():.1f} to {exp_energy_axis.max():.1f} eV")

                    # Find M5 peak in experimental data (search window 880-890 eV)
                    m5_window = (exp_energy_axis >= 880) & (exp_energy_axis <= 890)
                    print(f"  M5 search window (880-890 eV): {np.sum(m5_window)} channels")
                    if np.sum(m5_window) > 5:
                        m5_region_energy = exp_energy_axis[m5_window]
                        m5_region_data = summed_spectrum_norm[m5_window]
                        
                        print(f"  M5 region intensity range: {np.min(m5_region_data):.3f} to {np.max(m5_region_data):.3f}")

                        # Find peaks : ADJUST THRESHOLDS HERE IF NEEDED!! SET BASED ON YOUR DATA PARAMETERS.
                        peaks_m5, props_m5 = find_peaks(
                            m5_region_data,
                            height=0.08,
                            prominence=0.03,
                            distance=5
                        )
                        
                        print(f"  Found {len(peaks_m5)} candidate peaks in M5 region")

                        if len(peaks_m5) > 0:
                            # tallest peak as M5
                            tallest_idx = np.argmax(props_m5['peak_heights'])
                            exp_m5_idx = peaks_m5[tallest_idx]
                            is_valid, width_ev = validate_peak_width(
                                m5_region_data, exp_m5_idx, m5_region_energy, min_width_ev=0.5
                            )

                            if is_valid:
                                exp_m5_pos = float(m5_region_energy[exp_m5_idx])

                                # Get Ce4+ reference M5 position (should be near 883 eV nominally)
                                ref_ce4_energy = refs['ce4'].axes_manager[0].axis
                                ref_ce4_data = refs['ce4'].data
                                ref_ce4_norm = normalise_spectra(ref_ce4_data)

                                # Find reference M5 peak
                                ref_m5_window = (ref_ce4_energy >= 878) & (ref_ce4_energy <= 888)
                                if np.sum(ref_m5_window) > 5:
                                    ref_m5_region_energy = ref_ce4_energy[ref_m5_window]
                                    ref_m5_region_data = ref_ce4_norm[ref_m5_window]

                                    ref_peaks_m5, ref_props_m5 = find_peaks(ref_m5_region_data, height=0.3, prominence=0.1)

                                    if len(ref_peaks_m5) > 0:
                                        ref_m5_idx = ref_peaks_m5[np.argmax(ref_props_m5['peak_heights'])]
                                        ref_m5_pos = float(ref_m5_region_energy[ref_m5_idx])
                                        energy_shift = exp_m5_pos - ref_m5_pos

                                        print(f"  Experimental M5 peak found at: {exp_m5_pos:.2f} eV (width: {width_ev:.1f} eV)")
                                        print(f"  Ce4+ reference M5 peak at: {ref_m5_pos:.2f} eV")
                                        print(f"  Calculated shift to align references: {energy_shift:+.2f} eV")

                                        # Set to reasonable range - no false detects causing large movement
                                        max_shift = params.get('max_auto_shift_ev', 6.0)
                                        if abs(energy_shift) > max_shift:
                                            print(f"  WARNING: Shift {energy_shift:+.2f} eV exceeds maximum {max_shift:.1f} eV, clamping...")
                                            energy_shift = np.clip(energy_shift, -max_shift, max_shift)

                                        # Ignore tiny shifts/peak finding fails
                                        if abs(energy_shift) < 0.3:
                                            print(f"  Shift too small ({energy_shift:+.2f} eV < 0.3 eV), skipping")
                                            energy_shift = 0.0
                                    else:
                                        print("  Could not find M5 peak in Ce4+ reference")
                                        energy_shift = 0.0
                                else:
                                    print("  Insufficient data in reference M5 region")
                                    energy_shift = 0.0
                            else:
                                print(f"  Found peak at {m5_region_energy[exp_m5_idx]:.1f} eV but width {width_ev:.1f} eV too narrow (<1.5 eV)")
                                energy_shift = 0.0
                        else:
                            print("  No M5 peak found in experimental data")
                            energy_shift = 0.0
                    else:
                        print("  Insufficient data in M5 region (880-890 eV)")
                        energy_shift = 0.0

                except Exception as e:
                    print(f"  Energy shift estimation failed: {e}")
                    traceback.print_exc()
                    energy_shift = 0.0

            if abs(energy_shift) > 0.01:
                print(f"\nApplying energy shift to REFERENCES ONLY (Ce3+ and Ce4+ as a unit): {energy_shift:+.2f} eV")
                print(f"  Before: Ce3+ offset = {refs['ce3'].axes_manager[0].offset:.2f} eV, Ce4+ offset = {refs['ce4'].axes_manager[0].offset:.2f} eV")
                refs['ce3'].axes_manager[0].offset += energy_shift
                refs['ce4'].axes_manager[0].offset += energy_shift
                print(f"  After:  Ce3+ offset = {refs['ce3'].axes_manager[0].offset:.2f} eV, Ce4+ offset = {refs['ce4'].axes_manager[0].offset:.2f} eV")
                print(f"  Experimental data unchanged")
            else:
                print(f"\nNo energy shift applied (calculated shift: {energy_shift:+.2f} eV)")

            # Free LL from memory
            try:
                if s_LL is not None and ll_path is not None:
                    del s_LL
                    s_LL = None
                    gc.collect()
            except Exception:
                pass

            # Crop match reference spectra range (870.0 - 913.9 eV)
            e_min_target = params.get('eels_energy_min', 870.0)
            e_max_target = params.get('eels_energy_max', 913.9)
            # get energy axis from axes_manager, if not from shape
            try:
                energy_axis = s_bg.axes_manager[2].axis
            except Exception:
                energy_axis = np.arange(s_bg.data.shape[2])

            energy_axis = np.asarray(energy_axis)

            # Expand selection range slightly to account for different pixel sizes, energy scales, or other reasons for offsets
            energy_scale = float(s_bg.axes_manager[2].scale)
            margin = 2.0 * energy_scale 
            
            e_min_search = e_min_target - margin
            e_max_search = e_max_target + margin
            
            # Find indices of the target range (with margin)
            idx_min = int(np.searchsorted(energy_axis, e_min_search, side='left'))
            idx_max = int(np.searchsorted(energy_axis, e_max_search, side='right'))
            
            idx_min = max(0, idx_min)
            idx_max = min(energy_axis.size, idx_max)
            
            energy_selected = energy_axis[idx_min:idx_max]
            data_selected = s_bg.data[:, :, idx_min:idx_max]
            
            print(f"Selected {energy_selected.size} channels covering {energy_selected[0]:.1f}-{energy_selected[-1]:.1f} eV")
            print(f"  Target range: {e_min_target:.1f}-{e_max_target:.1f} eV (with {margin:.2f} eV margin)")
            
            # Validate coverage of reference spectra range after cropping
            if energy_selected[0] > e_min_target + 0.5 or energy_selected[-1] < e_max_target - 0.5:
                print(f"  WARNING: Selected range doesn't fully cover target!")
                print(f"    Missing at start: {max(0, e_min_target - energy_selected[0]):.1f} eV")
                print(f"    Missing at end: {max(0, energy_selected[-1] - e_max_target):.1f} eV")

            # Construct new energy axis to match reference spectral size EXACTLY, as it must match reference dimensions for NNLS/MLLS fitting
            target_ns = int(target_spectral_size)
            
            ref_offset = float(refs['ce3'].axes_manager[0].offset)  # Should be 870.0 eV
            ref_scale = float(refs['ce3'].axes_manager[0].scale)    # Should be 0.18 eV/channel
            new_e0 = ref_offset
            new_e1 = ref_offset + (target_ns - 1) * ref_scale  # Should be 913.9 eV
            
            print(f"Creating energy axis to match reference spectra (required for NNLS/MLLS):")
            print(f"  Target range: {new_e0:.1f} to {new_e1:.1f} eV ({target_ns} channels)")
            print(f"  Target scale: {ref_scale:.3f} eV/ch")
            print(f"  Source data available: {energy_selected[0]:.1f} to {energy_selected[-1]:.1f} eV ({len(energy_selected)} channels)")
            print(f"  Interpolation: {len(energy_selected)} source points changed to {target_ns} target points on exact reference grid")
            
            new_energy_axis = np.linspace(new_e0, new_e1, target_ns)   

            # Interpolate spectra to new_energy_axis
            h0, w0, ns0 = data_selected.shape
            data_flat = data_selected.reshape((h0 * w0, ns0))
            data_resampled = np.empty((data_flat.shape[0], target_ns), dtype=float)
            
            print(f"Interpolating to exact reference grid: {new_e0:.1f}-{new_e1:.1f} eV, {target_ns} channels")
            
            for r in range(data_flat.shape[0]):
                try:
                    data_resampled[r, :] = np.interp(new_energy_axis, energy_selected, data_flat[r, :])
                except Exception:
                    print(f"  Warning: Interpolation failed for pixel {r}, setting to zero")
                    data_resampled[r, :] = np.zeros(target_ns)

            # Detect sharp spikes on the summed spectrum (these can be introduced by RL)
            try:
                energy_new = new_energy_axis
                summed_pix = np.sum(data_resampled, axis=0)
                _, clean_sum, peaks = clean_and_normalise_spectrum(energy_new, summed_pix, prominence=0.05, width=(1,5), pad=1) # putting pad=1 to just normalise, CHANGE THIS AS YOU NEED TO BEST REMOVE ARTEFACTS
                bad = np.zeros_like(summed_pix, dtype=bool)
                all_peaks = np.concatenate([peaks.get('pos', np.array([], dtype=int)), peaks.get('neg', np.array([], dtype=int))])
                for p in all_peaks:
                    lo = max(0, int(p - 3))
                    hi = min(target_ns, int(p + 3 + 1))
                    bad[lo:hi] = True

                if np.any(bad) and np.sum(~bad) >= 2:
                    good = ~bad
                    # Interpolate across energy channels for removed sharp spikes
                    for r in range(data_resampled.shape[0]):
                        spec = data_resampled[r, :]
                        try:
                            data_resampled[r, bad] = np.interp(energy_new[bad], energy_new[good], spec[good])
                        except Exception:
                            pass
            except Exception:
                pass

            data_resampled = data_resampled.reshape((h0, w0, target_ns))
            s = hs.signals.Signal1D(data_resampled, metadata=s_bg.metadata.as_dictionary(), signal_type='EELS')
            try:
                s.axes_manager[2].scale = float(new_energy_axis[1] - new_energy_axis[0])
                s.axes_manager[2].offset = float(new_energy_axis[0])
                s.axes_manager[2].units = 'eV'
            except Exception:
                pass
            
            print(f"\nFinal processed signal:")
            print(f"  Shape: {s.data.shape}")
            print(f"  Energy range: {s.axes_manager[2].axis.min():.1f} - {s.axes_manager[2].axis.max():.1f} eV")
            print(f"  Channels: {len(s.axes_manager[2].axis)}")
            print(f"  Offset: {s.axes_manager[2].offset:.1f} eV")
            print(f"  Scale: {s.axes_manager[2].scale:.3f} eV/ch")
            print(f"  Matches reference: {len(s.axes_manager[2].axis) == target_spectral_size and abs(s.axes_manager[2].offset - ref_offset) < 0.01}")
            
            try:
                print("\nCreating summed EELS spectrum plot")
                summed_spectrum = np.sum(s.data, axis=(0, 1))
                
                # Apply improved smoothing routine
                summed_spectrum_clean = remove_spectral_spikes(summed_spectrum, threshold_sigma=2.0, window=9)
                summed_spectrum_clean2 = np.clip(summed_spectrum_clean, 0.0, None) 
                summed_spectrum_smoothed = uniform_filter1d(summed_spectrum_clean2, size=3)
                summed_spectrum_smoothed = gaussian_filter1d(summed_spectrum_smoothed, sigma=1.0)
                
                summed_spectrum_norm = normalise_spectra(summed_spectrum_smoothed)
                
                energy_axis_interp = s.axes_manager[2].axis
                
                # Find M5/M4 peaks with width validation
                m5_region = (energy_axis_interp >= 878) & (energy_axis_interp <= 888)
                if np.any(m5_region):
                    m5_peaks, _ = find_peaks(
                        summed_spectrum_norm[m5_region],
                        height=0.2,
                        prominence=0.1
                    )
                    
                    if len(m5_peaks) > 0:
                        valid_m5 = []
                        for idx in m5_peaks:
                            is_valid, width_ev = validate_peak_width(
                                summed_spectrum_norm[m5_region],
                                idx,
                                energy_axis_interp[m5_region],
                                min_width_ev=2.0
                            )
                            if is_valid:
                                valid_m5.append((idx, width_ev))
                        
                        if len(valid_m5) > 0:
                            m5_peak_idx, m5_width = valid_m5[0]
                            m5_peak_energy = energy_axis_interp[m5_region][m5_peak_idx]
                            m5_shift = m5_peak_energy - 883.0
                            print(f"  M5 peak at {m5_peak_energy:.2f} eV (width: {m5_width:.1f} eV, shift: {m5_shift:+.2f} eV)")
                        else:
                            print(f"  Warning: No valid M5 peak found (all peaks too narrow)")
                    else:
                        print(f"  Warning: No M5 peak detected in expected region")
                
                m4_region = (energy_axis_interp >= 896) & (energy_axis_interp <= 906)
                if np.any(m4_region):
                    m4_peaks, _ = find_peaks(
                        summed_spectrum_norm[m4_region],
                        height=0.2,
                        prominence=0.1
                    )
                    
                    if len(m4_peaks) > 0:
                        valid_m4 = []
                        for idx in m4_peaks:
                            is_valid, width_ev = validate_peak_width(
                                summed_spectrum_norm[m4_region],
                                idx,
                                energy_axis_interp[m4_region],
                                min_width_ev=2.0
                            )
                            if is_valid:
                                valid_m4.append((idx, width_ev))
                        
                        if len(valid_m4) > 0:
                            m4_peak_idx, m4_width = valid_m4[0]
                            m4_peak_energy = energy_axis_interp[m4_region][m4_peak_idx]
                            m4_shift = m4_peak_energy - 901.0
                            print(f"  M4 peak at {m4_peak_energy:.2f} eV (width: {m4_width:.1f} eV, shift: {m4_shift:+.2f} eV)")
                
                fig, ax = plt.subplots(figsize=(11, 7))
                ax.plot(energy_axis_interp, summed_spectrum_norm, 'k-', linewidth=2.5, label='Summed EELS Spectrum')

                ce3_ref_norm = normalise_spectra(np.array(refs['ce3'].data).astype(float))
                ce4_ref_norm = normalise_spectra(np.array(refs['ce4'].data).astype(float))
                ax.plot(refs['ce3'].axes_manager[0].axis, ce3_ref_norm, 'r--', linewidth=2, alpha=0.7, label=r'Ce$^{3+}$ ')
                ax.plot(refs['ce4'].axes_manager[0].axis, ce4_ref_norm, 'b--', linewidth=2, alpha=0.7, label=r'Ce$^{4+}$ ')
                
                # Mark peak positions
                ax.axvline(x=883, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
                ax.text(883, 1.05, r'M$_5$', ha='center', fontsize=11, color='gray')
                ax.axvline(x=901, color='gray', linestyle=':', alpha=0.4, linewidth=1.5)
                ax.text(901, 1.05, r'M$_4$', ha='center', fontsize=11, color='gray')
                
                ax.set_xlabel('Energy Loss (eV)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Normalised Intensity', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(energy_axis_interp.min(), energy_axis_interp.max())
                ax.set_ylim(-0.05, 1.1)
                
                plt.tight_layout()
                fig_path = os.path.join(figures_dir, f"{base_name}_summed_interpolated_spectrum.png")
                # Summed EELS spectrum
                save_figure(fig, fig_path)
                saved_figures.append(fig_path)
                # Summed EELS spectrum arrays
                np.savez(os.path.join(out_dir, f"{base_name}_summed_interpolated_spectrum_data.npz"),
                         energy_axis=energy_axis_interp,
                         spectrum_normalized=summed_spectrum_norm,
                         spectrum_spike_removed=summed_spectrum_clean)
                print(f"Saved summed spectrum plot: {fig_path}")
            except Exception as e:
                print(f"Warning: Could not create summed spectrum plot: {e}")
                traceback.print_exc()
                        
            # Validate sliced data has correct dimensions
            if len(s.data.shape) != 3:
                raise ValueError(f"Sliced data should be 3D (y, x, energy), got shape: {s.data.shape}")
            
            # crop to even dimensions for bin2 compatibility
            current_shape = s.data.shape
            even_y = current_shape[0] - (current_shape[0] % 2)
            even_x = current_shape[1] - (current_shape[1] % 2)
            
            print(f"Original shape: {current_shape}, cropping to even: ({even_y}, {even_x}, {current_shape[2]})")
            cropped_data = s.data[:even_y, :even_x, :]
            s_orig = hs.signals.Signal1D(cropped_data, metadata=s.metadata.as_dictionary(), signal_type='EELS')
            s_orig.axes_manager[2].scale = s.axes_manager[2].scale
            s_orig.axes_manager[2].offset = s.axes_manager[2].offset
            s_orig.axes_manager[2].units = 'eV'
            print(f"Original cropped shape: {s_orig.data.shape}")
            
            # Binned by 2 in spatial dimensions using block_reduce
            # s_orig is already cropped to even dimensions
            bin2_y = s_orig.data.shape[0] // 2
            bin2_x = s_orig.data.shape[1] // 2
            
            print(f"Bin2 calculated shape: ({bin2_y}, {bin2_x}, {s_orig.data.shape[2]})")
            # Use block_reduce on the cropped data (already even dimensions)
            data_bin2 = block_reduce(s_orig.data, block_size=(2, 2, 1), func=np.mean)
            print(f"Bin2 array shape after block_reduce: {data_bin2.shape}")
            
            # Verify bin2 shape matches expectation
            assert data_bin2.shape == (bin2_y, bin2_x, s_orig.data.shape[2]), f"Bin2 shape mismatch: {data_bin2.shape} vs expected ({bin2_y}, {bin2_x}, {s_orig.data.shape[2]})"
            
            # set up new Hyperspy signal
            s_bin2 = hs.signals.Signal1D(data_bin2, metadata=s_orig.metadata.as_dictionary(), signal_type='EELS')
            s_bin2.axes_manager[2].scale = s_orig.axes_manager[2].scale
            s_bin2.axes_manager[2].offset = s_orig.axes_manager[2].offset
            s_bin2.axes_manager[2].units = 'eV'
            print(f"Bin2 signal shape: {s_bin2.data.shape}")
            
            processing_versions.append({
                'name': 'bin2',
                'suffix': '_bin2',
                'data': s_bin2,
                'new_shape': data_bin2.shape,
                'rl_applied': True
            })

            # Process noRL version if available
            if s_bg_noRL is not None:
                print("\n" + "="*60)
                print("Processing non-RL version (background removed only)")
                print("="*60)
                
                # Process s_bg_noRL through same interpolation pipeline
                # Crop to match reference spectra range
                try:
                    energy_axis_noRL = np.asarray(s_bg_noRL.axes_manager[2].axis)
                except Exception:
                    energy_axis_noRL = np.arange(s_bg_noRL.data.shape[2])
                
                # Find indices of the target range (with margin)
                idx_min_noRL = int(np.searchsorted(energy_axis_noRL, e_min_search, side='left'))
                idx_max_noRL = int(np.searchsorted(energy_axis_noRL, e_max_search, side='right'))
                
                idx_min_noRL = max(0, idx_min_noRL)
                idx_max_noRL = min(energy_axis_noRL.size, idx_max_noRL)
                
                energy_selected_noRL = energy_axis_noRL[idx_min_noRL:idx_max_noRL]
                data_selected_noRL = s_bg_noRL.data[:, :, idx_min_noRL:idx_max_noRL]
                
                print(f"NoRL: Selected {energy_selected_noRL.size} channels covering {energy_selected_noRL[0]:.1f}-{energy_selected_noRL[-1]:.1f} eV")
                
                # Interpolate to reference grid
                h0_noRL, w0_noRL, ns0_noRL = data_selected_noRL.shape
                data_flat_noRL = data_selected_noRL.reshape((h0_noRL * w0_noRL, ns0_noRL))
                data_resampled_noRL = np.empty((data_flat_noRL.shape[0], target_ns), dtype=float)
                
                print(f"NoRL: Interpolating to exact reference grid: {new_e0:.1f}-{new_e1:.1f} eV, {target_ns} channels")
                
                for r in range(data_flat_noRL.shape[0]):
                    try:
                        data_resampled_noRL[r, :] = np.interp(new_energy_axis, energy_selected_noRL, data_flat_noRL[r, :])
                    except Exception:
                        data_resampled_noRL[r, :] = np.zeros(target_ns)
                
                data_resampled_noRL = data_resampled_noRL.reshape((h0_noRL, w0_noRL, target_ns))
                
                # Create Hyperspy Signal1D
                s_noRL = hs.signals.Signal1D(data_resampled_noRL, metadata=s_bg_noRL.metadata.as_dictionary(), signal_type='EELS')
                s_noRL.axes_manager[2].scale = float(new_energy_axis[1] - new_energy_axis[0])
                s_noRL.axes_manager[2].offset = float(new_energy_axis[0])
                s_noRL.axes_manager[2].units = 'eV'
                
                print(f"NoRL: Final processed signal shape: {s_noRL.data.shape}")
                
                # Crop to even dimensions and bin spatially
                current_shape_noRL = s_noRL.data.shape
                even_y_noRL = current_shape_noRL[0] - (current_shape_noRL[0] % 2)
                even_x_noRL = current_shape_noRL[1] - (current_shape_noRL[1] % 2)
                
                cropped_data_noRL = s_noRL.data[:even_y_noRL, :even_x_noRL, :]
                s_orig_noRL = hs.signals.Signal1D(cropped_data_noRL, metadata=s_noRL.metadata.as_dictionary(), signal_type='EELS')
                s_orig_noRL.axes_manager[2].scale = s_noRL.axes_manager[2].scale
                s_orig_noRL.axes_manager[2].offset = s_noRL.axes_manager[2].offset
                s_orig_noRL.axes_manager[2].units = 'eV'
                
                # Bin 2x2
                data_bin2_noRL = block_reduce(s_orig_noRL.data, block_size=(2, 2, 1), func=np.mean)
                
                s_bin2_noRL = hs.signals.Signal1D(data_bin2_noRL, metadata=s_orig_noRL.metadata.as_dictionary(), signal_type='EELS')
                s_bin2_noRL.axes_manager[2].scale = s_orig_noRL.axes_manager[2].scale
                s_bin2_noRL.axes_manager[2].offset = s_orig_noRL.axes_manager[2].offset
                s_bin2_noRL.axes_manager[2].units = 'eV'
                
                print(f"NoRL: Bin2 signal shape: {s_bin2_noRL.data.shape}")
                
                processing_versions.append({
                    'name': 'bin2_noRL',
                    'suffix': '_bin2_noRL',
                    'data': s_bin2_noRL,
                    'new_shape': data_bin2_noRL.shape,
                    'rl_applied': False
                })


        # NNLS/MLLS fitting for each version
        for version in processing_versions:
            print(f"\nProcessing version: {version['name']}")
            eels_data = version['data']
            height, width, _ = eels_data.data.shape
            version_suffix = version['suffix']
            pixel_size = 0.3 if version['name'] == 'bin2' else 0.15
            
            A = np.vstack([refs['ce3'].data, refs['ce4'].data]).T
            nnls_args = []
            for i in range(height):
                for j in range(width):
                    # Treat all pixels as signal (no background masking)
                    nnls_args.append((i, j, eels_data.data[i, j], A, None, False))

            ce3_map = np.zeros((height, width))
            ce4_map = np.zeros((height, width))
            ce3_mlls = np.zeros((height, width))
            ce4_mlls = np.zeros((height, width))

            with Pool(processes=(nproc or max(1, cpu_count()-1))) as pool:
                for res in tqdm(pool.imap_unordered(nnls_worker, nnls_args), total=len(nnls_args)):
                    i, j, nnls_c3, nnls_c4, mlls_c3, mlls_c4 = res
                    ce3_map[i, j] = nnls_c3
                    ce4_map[i, j] = nnls_c4
                    ce3_mlls[i, j] = mlls_c3
                    ce4_mlls[i, j] = mlls_c4

            # for bin2 only save NNLS and MLLS arrays
            try:
                if version['name'] == 'bin2':
                    # RL version: save Ce3 maps only 
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce3_map_nnls.npy"), ce3_map)
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce3_map_mlls.npy"), ce3_mlls)
                    print(f"  Saved RL Ce3 maps")
                elif version['name'] == 'bin2_noRL':
                    # NoRL version: save both Ce3 and Ce4 maps for comparison, good for debugging
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce3_map_nnls.npy"), ce3_map)
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce3_map_mlls.npy"), ce3_mlls)
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce4_map_nnls.npy"), ce4_map)
                    np.save(os.path.join(out_dir, f"{base_name}{version_suffix}_ce4_map_mlls.npy"), ce4_mlls)
                    print(f"  Saved noRL Ce3 and Ce4 maps")
                else:
                    # For non-bin2 versions only keep in-memory results (no file writes - can change this here if saved needed!)
                    pass
            except Exception as e:
                print(f"  Warning: Could not save maps: {e}")
            continue

        return True
        
    except Exception as e:
        print(f"\nProcessing failed: {type(e).__name__}")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False

def find_hspy_pairs(input_dir):
    """
    Find complete EELS datasets based on summed files.
    
    Parameters:
        input_dir : str
            Input directory to search
    
    Returns:
        list
            List of paths to HL summed files
    """
    results = []
    seen_bases = set()
    
    # Match files with _HL_ and summing suffixes
    for root, dirs, files in os.walk(input_dir):
        # Exclude output/processed directories from search
        dirs[:] = [d for d in dirs if 'processed' not in d.lower() and 'output' not in d.lower()]
        
        hspy_files = [f for f in files 
                     if f.lower().endswith('.hspy') 
                     and '_hl_' in f.lower()
                     and '_sum' in f.lower() 
                     and '_processed' not in f.lower()]

        for f in hspy_files:
            # Keep the FULL filename as the base
            # ensures each variant is processed separately
            base = os.path.splitext(f)[0]
            seen_bases.add((root, f))
    
    print(f"Found {len(seen_bases)} potential datasets")
    
    # find related files for each summing variant
    for root, hl_filename in seen_bases:
        hl_full_path = os.path.join(root, hl_filename)
        
        import re
        hl_lower = hl_filename.lower()
        base = hl_lower.replace('.hspy', '')
        sum_variant = None
        
        # Extract sum variant first
        if '_sumall_' in hl_lower:
            match = re.search(r'_sumall_(\d+)', hl_lower)
            if match:
                sum_variant = f'sumall_{match.group(1)}'
        elif '_sum' in hl_lower:
            match = re.search(r'_sum(\d+)', hl_lower)
            if match:
                sum_variant = f'sum{match.group(1)}'
        
        # Remove HL/LL suffix and sum variant to get base identifier
        pattern = r'_(hl|ll)_stack(_sum(all_)?\d+)?$'
        match = re.search(pattern, base, re.IGNORECASE)
        if match:
            base = base[:match.start()]
        
        dataset_files = {'HL_summed': hl_full_path}
        
        # Find companion files (LL, ADF, EDS) with matching sum variant
        for f in os.listdir(root):
            f_lower = f.lower()
            if base not in f_lower or not f_lower.endswith('.hspy'):
                continue

            full_path = os.path.join(root, f)

            # Skip already processed files
            skip_keywords = ['_processed', '_edge', '_bulk', '_deconvolved', '_nnls', '_mlls', '_classification', '_cleaned']
            if any(keyword in f_lower for keyword in skip_keywords):
                continue

            # Match LL with same sum variant
            if '_ll' in f_lower and sum_variant:
                if sum_variant in f_lower:
                    dataset_files['LL_summed'] = full_path
            elif ('_adf' in f_lower) or ('haadf' in f_lower) or ('hadf' in f_lower):
                if 'ADF' not in dataset_files:  
                    dataset_files['ADF'] = full_path
            elif '_eds' in f_lower:
                if 'EDS' not in dataset_files: 
                    dataset_files['EDS'] = full_path
        
        # Add this dataset
        if os.path.exists(hl_full_path):
            results.append(hl_full_path)
            print(f"\nFound dataset: {base}")
            for k, v in dataset_files.items():
                print(f"  {k}: {os.path.basename(v)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch EELS/ADF/EDS processing v3')
    parser.add_argument('--input_dir', required=True, help='Input directory with .hspy files')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--ref_ce3', required=True, help='Path to Ce3+ reference spectrum')
    parser.add_argument('--ref_ce4', required=True, help='Path to Ce4+ reference spectrum')
    parser.add_argument('--nproc', type=int, default=None, help='Number of processes')
    parser.add_argument('--eels_scale', type=float, default=0.1795, help='EELS energy scale (eV/channel)')
    parser.add_argument('--eels_offset', type=float, default=314, help='EELS energy offset (eV)')
    parser.add_argument('--ll_offset_correction', type=float, default=None, 
                        help='LL to HL energy offset correction in eV. Applied to LL before alignment.')
    parser.add_argument('--energy_shift_correction', type=float, default=0.0,
                        help='Energy shift correction in eV to align experimental peaks with references to improve MLLS/ NNLS fitting (default: 0.0). Positive shifts data to higher energies.')
    parser.add_argument('--skip_rl', action='store_true',
                        help='Skip Richardson-Lucy deconvolution even when LL is available')
    parser.add_argument('--preserve_subfolders', action='store_true',
                        help='Preserve input subfolder structure in output (default: False)')
    args = parser.parse_args()
    
    ensure_dir(args.out_dir)
    
    # Load references
    ce3 = hs.load(args.ref_ce3)
    ce4 = hs.load(args.ref_ce4)
    refs = {'ce3': ce3, 'ce4': ce4}
    
    print("REFERENCE SPECTRA INFORMATION")
    ref_m5_peak = None
    ref_m4_peak = None
    for name, ref in refs.items():
        try:
            energy_axis = ref.axes_manager[-1].axis
            e_min = energy_axis.min()
            e_max = energy_axis.max()
            n_channels = len(energy_axis)
            scale = ref.axes_manager[-1].scale
            offset = ref.axes_manager[-1].offset
            
            if name == 'ce4':
                ref_data = ref.data
                peaks, _ = find_peaks(ref_data, height=ref_data.max()*0.3, distance=50)
                if len(peaks) >= 2:
                    ref_m5_peak = energy_axis[peaks[0]]
                    ref_m4_peak = energy_axis[peaks[1]] if len(peaks) > 1 else None
            
            print(f"\n{name.upper()} reference:")
            print(f"  File: {args.ref_ce3 if name == 'ce3' else args.ref_ce4}")
            print(f"  Energy range: {e_min:.1f} - {e_max:.1f} eV")
            print(f"  Channels: {n_channels}")
            print(f"  Scale: {scale:.3f} eV/channel")
            print(f"  Offset: {offset:.1f} eV")
            if name == 'ce4' and ref_m5_peak:
                print(f"  M5 peak: approx {ref_m5_peak:.1f} eV")
                if ref_m4_peak:
                    print(f"  M4 peak: approx {ref_m4_peak:.1f} eV")
        except Exception as e:
            print(f"\n{name.upper()} reference: Could not read axis info ({e})")
            print(f"  Data shape: {ref.data.shape}")
        
    # Parameters
    params = {
        'eels_slice_idx': [3102, 3417],  # spectral slice indices for Ce edge (adjust per dataset if needed, main aim is to cover 970-915 approx with 245 channels to match refs )
        'eels_scale': args.eels_scale,
        'eels_offset': args.eels_offset,
        'll_offset_correction': args.ll_offset_correction,
        'energy_shift_correction': args.energy_shift_correction,
        'skip_rl': args.skip_rl,
        # Peak detection/internal defaults
        'expected_m5_ev': 885.0,
        'expected_m4_ev': 904.0,
        'ref_m5_ev': 883.0,
        'ref_m4_ev': 901.0,
        'peak_window_ev': 5.0,
        'max_auto_shift_ev': 6.0,
    }
    
    datasets = find_hspy_pairs(args.input_dir)
    print(f'Found {len(datasets)} datasets to process')
    
    for summed in datasets:
        # Extract filename and create subfolder based on prefix before "insitu"
        base_filename = os.path.basename(summed)
        
        if 'insitu' in base_filename.lower():
            insitu_pos = base_filename.lower().find('insitu')
            sample_prefix = base_filename[:insitu_pos].rstrip('_-')
        else:
            # use the base name before first underscore or if not the whole name
            sample_prefix = base_filename.split('_')[0] if '_' in base_filename else os.path.splitext(base_filename)[0]
        
        # Create output directory structure
        if args.preserve_subfolders:
            # Preserve input subfolder structure, e.g. out_dir/subfolder/sample_prefix/
            rel_path = os.path.relpath(summed, args.input_dir)
            rel_dir = os.path.dirname(rel_path)
            outsub = os.path.join(args.out_dir, rel_dir, sample_prefix)
            print(f'\nProcessing sample: {sample_prefix} (subfolder: {rel_dir})')
        else:
            # organise by sample prefix only: out_dir/sample_prefix/
            outsub = os.path.join(args.out_dir, sample_prefix)
            print(f'\nProcessing sample: {sample_prefix}')
        
        ensure_dir(outsub)
        print(f'  Input: {summed}')
        print(f'  Output: {outsub}')
        
        try:
            process_dataset_v3(summed, outsub, refs, params, nproc=args.nproc)
            print(f' Processed: {summed}')
        except Exception as e:
            print(f' Failed: {summed}\n  Error: {e}')
            # Print full traceback to identify exact failure line
            traceback.print_exc()
        finally:
            gc.collect()
            # memory cleanup as processing many datasets

if __name__ == "__main__":
    main()

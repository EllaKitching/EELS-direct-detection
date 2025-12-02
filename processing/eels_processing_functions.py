"""
Functions for EELS processing workflows

Utilities made for analysis of EELS spectrum image data, including:
* Mask creation from ADF images
* Edge spectrum extraction for surface analysis
* Spectrum artefact cleaning and normalisation
* Energy shift estimation via peak matching
* Multi linear least squares fitting

Author: Ella Kitching, Cardiff University, 2025
License: GPL-3.0
"""
import numpy as np

from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, median_filter, gaussian_filter
from scipy.signal import find_peaks

from skimage.filters import (
    threshold_triangle,
    threshold_otsu,
    threshold_li,
    threshold_yen,
    threshold_mean,
)
from skimage.morphology import remove_small_objects
from skimage.restoration import denoise_nl_means, estimate_sigma

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle

import hyperspy.api as hs

# mask and map post processing functions
def create_distance_masks(adf_data, threshold_value=None, threshold_method='li', existing_mask=None, pixel=6):
    """
    Create distance based masks from ADF image or use existing mask, calculating distance via euclidean transform.
    
    Parameters:
    adf_data : 2D ndarray
        ADF image data
    threshold_value : float or None
        Threshold value for binary mask creation. If None, use threshold_method.
    threshold_method : str
        Method for automatic thresholding if threshold_value is None. Options: 'otsu', 'li', 'triangle', 'yen', 'mean'
    existing_mask : 2D ndarray or None 
        If provided, use this binary mask instead of creating one for distance calculation.
    pixel: int
        Distance in pixels to define edge region.
    
    Returns:
        distance_map : 2D ndarray
            Distance map from the perimeter
        mask_edge : 2D ndarray
            Mask for edge regions, defined as within pixels of perimeter
        mask_bulk : 2D ndarray
            Mask for bulk regions, defined as beyond pixels from perimeter
        threshold_value : float or None
            Threshold value used for mask creation (None if existing_mask was used)
    """

    if existing_mask is not None:
        binary = existing_mask
        threshold_value = None
    elif threshold_value is None:
        threshold_methods = {
            'otsu': lambda x: threshold_otsu(x, nbins=3),
            'li': threshold_li,
            'triangle': threshold_triangle,
            'yen': threshold_yen,
            'mean': threshold_mean
        }
        threshold_func = threshold_methods.get(threshold_method, threshold_triangle)
        threshold_value = threshold_func(adf_data)
        threshold_value = int(threshold_value / 100) * 100
        binary = adf_data > threshold_value
    else:
        binary = adf_data > threshold_value
    
    perimeter = binary & (~binary_erosion(binary, structure=np.ones((3, 3))))
    
    h, w = perimeter.shape
    yy, xx = np.indices((h, w))
    dist_to_border = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx])
    border_clearance = 3
    perimeter[dist_to_border < border_clearance] = False
    
    try:
        perimeter = remove_small_objects(perimeter, min_size=2)
    except Exception:
        pass
    
    distance_to_perimeter = ndi.distance_transform_edt(~perimeter)
    distance_map = distance_to_perimeter.astype(float)
    distance_map[~binary] = np.inf
    
    mask_edge = (distance_map >= 0) & (distance_map <= pixel) & binary
    mask_bulk = (distance_map > pixel) & (distance_map <= np.inf) & binary
    
    return distance_map, mask_edge, mask_bulk, threshold_value

def extract_edge_spectrum(cumulative_cube_data, edge_mask, energy_axis, 
                          ref_energy_offset, ref_energy_high, ref_energy_axis):
    """
    Extract and normalise the mean spectrum for an edge region from a cumulative cube,
    and interpolate it onto a reference energy axis.

    Parameters:
        cumulative_cube_data : ndarray
            3D array (y, x, E) containing cumulative spectral data.
        edge_mask : 2D boolean ndarray
            Mask selecting edge pixels (True = include).
        energy_axis : 1D ndarray
            Energy axis corresponding to the E dimension of cumulative_cube_data.
        ref_energy_offset : float
            Lower bound (eV) of the energy window to consider.
        ref_energy_high : float
            Upper bound (eV) of the energy window to consider.
        ref_energy_axis : 1D ndarray
            Reference energy axis to interpolate the extracted spectrum onto.

    Returns:
        1D ndarray or None
            Normalised spectrum interpolated to ref_energy_axis (max = 1.0), or
            None if extraction fails (no pixels in mask or insufficient energy points).
    """
    if cumulative_cube_data.ndim != 3:
        raise ValueError("cumulative_cube_data must be (y,x,E)")
    edge_cube = cumulative_cube_data[edge_mask]
    if edge_cube.size == 0:
        return None
    spec = edge_cube.mean(axis=0)  # (E)
    sel = (energy_axis >= ref_energy_offset) & (energy_axis <= ref_energy_high)
    if sel.sum() < 5:
        return None
    spec_cropped = spec[sel]
    e_cropped = energy_axis[sel]
    spec_interp = np.interp(ref_energy_axis, e_cropped, spec_cropped)
    spec_interp = np.clip(spec_interp, 0, None)
    mv = spec_interp.max()
    return spec_interp / mv if mv > 0 else spec_interp

def smooth_map(im, method='gaussian', sigma=0.8, median_size=None, nlm=False):
    """
    Smooth or denoise an image using different methods.

    Parameters:
        im : 2D ndarray
            Input image.
        method : str, optional
            Smoothing method. Options:
             'gaussian' : gaussian_filter with sigma (pixels)
             'median'   : median_filter with kernel size median_size
             'nlm'      : non local means denoising (edge preserving, but slower)
             any other value returns the image unchanged
        sigma : float, optional
            Gaussian sigma in pixels (used when method == 'gaussian'), default 0.8.
        median_size : int or None, optional
            Kernel size for median filter (used when method == 'median'). If None, defaults to 3.
        nlm : bool, optional
            Flag indicating non local means processing (kept for API compatibility; method=='nlm' triggers NLM).

    Returns:
        out : 2D ndarray
            Smoothed/denoised image as float array.
    """
    imf = im.copy().astype(float)
    if method == 'gaussian':
        out = gaussian_filter(imf, sigma=sigma)
    elif method == 'median':
        if median_size is None:
            median_size = 3
        out = median_filter(imf, size=median_size)
    elif method == 'nlm':
        sigma_est = np.mean(estimate_sigma(imf, multichannel=False))
        patch_kw = dict(patch_size=3, patch_distance=5, multichannel=False)
        out = denoise_nl_means(imf, h=1.2 * sigma_est, fast_mode=True, **patch_kw)
    else:
        out = imf
    return out

# EELS spectrum processing functions
def clean_and_normalise_spectrum(energy, spectrum, prominence=0.05, width=(1,5), pad=3, neg_prominence=None, clip_negative=True):
    """
    Detect narrow positive spikes and negative dips, interpolate across them, then normalise.
    Parameters:
        energy, spectrum : 1D arrays (same length)
        prominence, width, pad : used for scipy.signal.find_peaks for positive spikes
        neg_prominence : if None, uses same as `prominence` for negative dips detection
        clip_negative : if True, clips any small negative values to 0 after interpolation
    Returns: 
        norm_spec
            normalised cleaned spectrum with spikes removed, 1D array
        clean_spec
            cleaned spectrum with spikes removed, 1D array
        peaks 
            peaks is {'pos': pos_indices, 'neg': neg_indices}
    """
    spec = np.array(spectrum, dtype=float)
    en = np.array(energy, dtype=float)

    # detect spikes
    pos_peaks, _ = find_peaks(spec, prominence=prominence, width=width)
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
        clean_spec = spec.copy()
    else:
        clean_spec = spec.copy()
        clean_spec[~mask] = np.interp(en[~mask], en[mask], spec[mask])

    if clip_negative:
        clean_spec = np.clip(clean_spec, 0.0, None)

    # Normalise by integrated area (safe against zero)
    area = np.trapz(clean_spec, en)
    norm_spec = clean_spec / area if area != 0 else clean_spec

    peaks = {'pos': pos_peaks, 'neg': neg_peaks}
    return norm_spec, clean_spec, peaks
    
def mask_energy_range(energy, spec, e_min=878.0, e_max=880.0):
    """
    Mask out a specific energy range in the spectrum and interpolate across it.
    Parameters:
        energy, spec : 1D arrays (same length)
        e_min, e_max : energy range to mask out
    Returns:
        spec_interp : spectrum with masked region interpolated, 1D array
    """
    mask = (energy < e_min) | (energy > e_max)
    spec_interp = spec.copy()
    spec_interp[~mask] = np.interp(energy[~mask], energy[mask], spec[mask])
    return spec_interp

def normalise_spectra(y_values):
    """
    Normalise y_values in spectrum to range [0, 1]
    """
    y_min, y_max = y_values.min(), y_values.max()
    norm = (y_values - y_min) / (y_max - y_min)
    return norm

def remove_spectral_spikes(spec, threshold_sigma=4.0, window=5):
    """
    Remove extreme positive and negative spikes from spectrum using median filtering
    
    Parameters:
    spec : ndarray, Input spectrum
    threshold_sigma : float
        Number of standard deviations to use as threshold (default: 4.0)
    window : int
        Window size for median filter (default: 5)
    
    Returns:
    cleaned : ndarray
        Spectrum with spikes removed
    """
    
    # Force computation if this is a lazy dask array, hspy often uses dask
    if hasattr(spec, 'compute'):
        spec = spec.compute()
    
    cleaned = spec.copy()
    filtered = median_filter(cleaned, size=window, mode='reflect')
    residuals = cleaned - filtered
    residual_std = np.std(residuals)
    threshold = threshold_sigma * residual_std
    spike_mask = np.abs(residuals) > threshold
    cleaned[spike_mask] = filtered[spike_mask]
    return cleaned

def cumulative_eels(frame_idx, hl_signal):
    """
    Create cumulative sum signal up to a given frame index.

    Parameters:
        frame_idx : int
            Frame index to sum up to (inclusive).
        hl_signal : hyperspy signal
            HL rebinned signal with the navigation axis representing frames.

    Returns:
        hyperspy signal
            Hyperspy signal containing the cumulative sum across frames.
    """
    return hl_signal.inav[:,:,:frame_idx].sum(axis=2)  


def compute_m5_shift(spectrum_norm, energy_axis, ce4_ref_arr, ce4_energy_arr,
                     search_exp=(880, 890), search_ref=(878, 888),
                     min_prom=0.1, min_height=0.3, min_width_ev=0.5,
                     max_shift_ev=6.0, ignore_below=0.3):
    """
    Estimate an energy shift by matching a peak (M5) in an experimental normalised
    spectrum to a reference CE4 spectrum.

    Parameters:
        spectrum_norm : 1D ndarray
            Normalised experimental spectrum.
        energy_axis : 1D ndarray
            Energy axis for the experimental spectrum (same length as spectrum_norm).
        ce4_ref_arr : 1D ndarray
            Reference CE4 spectrum intensities.
        ce4_energy_arr : 1D ndarray
            Energy axis for the CE4 reference.
        search_exp : tuple (low, high), optional
            Energy window (eV) to search for the experimental peak (default: (880, 890)).
        search_ref : tuple (low, high), optional
            Energy window (eV) to search for the reference peak (default: (878, 888)).
        min_prom : float, optional
            Minimum prominence for peak detection (default: 0.1).
        min_height : float, optional
            Minimum peak height for detection (default: 0.3).
        min_width_ev : float, optional
            Minimum peak full width at half maximum in eV (default: 0.5).
        max_shift_ev : float, optional
            Maximum allowed absolute shift in eV; shifts beyond this are clipped
            (default: 6.0).
        ignore_below : float, optional
            Return zero if estimated shift magnitude is smaller than this value (default: 0.3).

    Returns:
        float
            Estimated energy shift (exp_peak - ref_peak) in eV. Returns 0.0 when no
            reliable peak/shift is found.
    """
    # Helper function to validate peak width
    def validate_width(spec, pk_idx, e_ax):
        half = spec[pk_idx] / 2.0
        li = pk_idx
        while li > 0 and spec[li] > half:
            li -= 1
        ri = pk_idx
        while ri < len(spec) - 1 and spec[ri] > half:
            ri += 1
        return abs(e_ax[ri] - e_ax[li])

    exp_mask = (energy_axis >= search_exp[0]) & (energy_axis <= search_exp[1])
    if exp_mask.sum() < 5:
        return 0.0
    exp_region_e = energy_axis[exp_mask]
    exp_region_s = spectrum_norm[exp_mask]
    pks, props = find_peaks(exp_region_s, prominence=min_prom, height=min_height, distance=5)
    if len(pks) == 0:
        return 0.0
    pk = pks[np.argmax(props['peak_heights'])]
    width = validate_width(exp_region_s, pk, exp_region_e)
    if width < min_width_ev:
        return 0.0
    exp_m5 = float(exp_region_e[pk])

    ce4_norm = ce4_ref_arr / max(ce4_ref_arr.max(), 1e-12)
    ref_mask = (ce4_energy_arr >= search_ref[0]) & (ce4_energy_arr <= search_ref[1])
    if ref_mask.sum() < 5:
        return 0.0
    ref_region_e = ce4_energy_arr[ref_mask]
    ref_region_s = ce4_norm[ref_mask]
    rpks, rprops = find_peaks(ref_region_s, prominence=min_prom, height=min_height)
    if len(rpks) == 0:
        return 0.0
    rpk = rpks[np.argmax(rprops['peak_heights'])]
    ref_m5 = float(ref_region_e[rpk])

    shift = exp_m5 - ref_m5
    if abs(shift) > max_shift_ev:
        shift = np.clip(shift, -max_shift_ev, max_shift_ev)
    if abs(shift) < ignore_below:
        return 0.0
    return shift

def mlls_fit(spec_norm, r3, r4):
    """
    Perform a multi linear least squares (MLLS) fit using two reference spectra.

    Parameters:
        spec_norm : 1D ndarray or None
            Target spectrum to fit (same length as r3 and r4). If None, function returns zeros/nan.
        r3 : 1D ndarray
            Reference spectrum 3 (same length as spec_norm).
        r4 : 1D ndarray
            Reference spectrum 4 (same length as spec_norm).

    Returns:
        tuple (c3, c4, frac, ratio)
            c3 : float
                Fitted coefficient for r3 (clipped to >= 0).
            c4 : float
                Fitted coefficient for r4 (clipped to >= 0).
            frac : float
                Fractional contribution of r3 (c3 / (c3 + c4)) or 0.0 if total is zero.
            ratio : float or nan
                Ratio c3 / c4, or np.nan if c4 == 0.
    """
    if spec_norm is None:
        return 0.0, 0.0, 0.0, np.nan
    A = np.vstack([r3, r4]).T
    if A.shape[0] != spec_norm.size:
        raise ValueError("Spectrum length mismatch with references")
    coeffs, _, _, _ = np.linalg.lstsq(A, spec_norm, rcond=None)
    c3 = max(coeffs[0], 0.0)
    c4 = max(coeffs[1], 0.0)
    tot = c3 + c4
    frac = c3 / tot if tot > 0 else 0.0
    ratio = c3 / c4 if c4 > 0 else np.nan
    return c3, c4, frac, ratio

# Matplotlib related functions
def save_figure(fig, outpath, dpi=200, tight=True, pad_inches=0.02):
    """
    Save a matplotlib figure with optional tight bounding box cropping.

    Parameters:
        fig: matplotlib.figure.Figure
        outpath: str, output filename
        dpi: int, resolution
        tight: bool, when True use bbox_inches='tight' to trim whitespace. For image
             maps (e.g., MLLS/NNLS), you may want to set to False to avoid any chance of edge clipping.
        pad_inches: float, padding used 
    Returns:
        None, but saves figure to outpath
    """
    try:
        if tight:
            fig.savefig(outpath, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi)
        else:
            fig.savefig(outpath, dpi=dpi)
    except ValueError as e:
        if "Image size" in str(e) and "too large" in str(e):
            # bbox_inches='tight' sometimes miscalculates, retry without it
            print(f"Warning: bbox_inches='tight' caused error, saving without it for {outpath}")
            fig.savefig(outpath, dpi=dpi)
        else:
            raise
    finally:
        plt.close(fig)


def add_scalebar(ax, pixel_size_nm=0.15, scale_length_nm=5, location='lower left', 
                 color='white', fontsize=12, linewidth=3, outlinewidth=3):
    """
    Add a scale bar to an image axis in a consistent manner.
    Drawn as a filled rectangle with a black edge so all four sides are outlined.

    Parameters:
        ax : matplotlib.axes.Axes
            Axis to draw the scale bar on.
        pixel_size_nm : float, optional
            Size of one pixel in nanometres (default: 0.15).
        scale_length_nm : float, optional
            Desired length of the scale bar in nanometres (default: 5).
        location : str, optional
            Location of the scale bar, e.g. 'lower left', 'lower right',
            'upper left', 'upper right' (default: 'lower left').
        color : str or tuple, optional
            Colour of the scale bar and label text (default: 'white').
        fontsize : int, optional
            Font size for the label in points (default: 12).
        linewidth : float or int, optional
            Thickness of the scale bar in points (default: 3).
        outlinewidth : float or int, optional
            Width of the black outline stroke around the bar in points (default: 3).

    Returns:
        None, but adds scale bar to the provided axis. Check figure.
    """

    scale_pixels = scale_length_nm / pixel_size_nm

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    img_width = xlim[1] - xlim[0]
    img_height = ylim[0] - ylim[1]

    if 'left' in location.lower():
        x_offset = img_width * 0.03
    else:
        x_offset = img_width * 0.80

    if 'lower' in location.lower():
        y_center = img_height * 0.95
    else:
        y_center = img_height * 0.05

    # Convert linewidth in points to data units (pixels) using figure DPI.
    # I used DPI because matplotlib's internal conversion is based on that, 1000 DPI typically used for high res figures.
    try:
        dpi = ax.figure.dpi
    except Exception:
        dpi = 100.0
    bar_thickness_px = max(1.0, (linewidth) * (dpi / 72.0))  # 1 pt = 1/72 inch

    # Draw rectangle (aka the scale bar) with black outline on all sides
    try:
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
    except Exception:
        # two line method (black underlay + white overlay) if needed
        ax.plot([x_offset, x_offset + scale_pixels], [y_center, y_center],
                color='black', linewidth=linewidth + 4, solid_capstyle='projecting', zorder=2)
        ax.plot([x_offset + 1, x_offset + scale_pixels - 1], [y_center, y_center],
                color=color, linewidth=linewidth, solid_capstyle='butt', zorder=3)

    # Label
    text_x = x_offset + scale_pixels / 2.0
    text_y = y_center - img_height * 0.015
    text = ax.text(text_x, text_y, f'{scale_length_nm:.0f} nm',
                   color=color, fontsize=fontsize, ha='center', va='bottom',
                   fontweight='bold')
    try:
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='black'),
            path_effects.Normal()
        ])
    except Exception:
        pass

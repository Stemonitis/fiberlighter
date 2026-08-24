"""Denoising and smoothing.

Each function applies one method to a single channel (gcamp by default) of
every animal and returns the same dict. All modify in place.

Applied to gcamp only because iso is a reference channel — smoothing it
changes the artifact estimate it exists to provide. Pass key="iso" if you
have a reason to.

TODO / known issues
-------------------
- Order matters: denoise before or after bleaching correction gives different
  results. Decide and document.
- wavelet_denoise attenuates transients hardest (tested: recovered a 2.0
  transient as 1.4, vs 2.2 for savgol and median). threshold_scale below 1.0
  is gentler. Check against your own data before trusting the default.
- savgol window=11 at ~3 Hz is a 3.7 s window — wide relative to GCaMP
  transients. Window is in SAMPLES, not seconds, so it means different things
  at different acquisition rates.
- medfilt kernel must be odd; even values raise.
- At 3 Hz most of these are marginal for the same reason lowpass was.
- No provenance, mutates in place — same as the bleaching module.
"""

import numpy as np
import pywt
from scipy.signal import savgol_filter, medfilt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d, median_filter as _medfilt_nd

def lowpass_filter(data, key="gcamp", cutoff=1.0, fs=None, order=4):
    """Butterworth lowpass. fs is derived from timestamps if None.

    Rarely useful at ~3 Hz acquisition: Nyquist is 1.5 Hz, and GCaMP
    transients have their energy in the same band you would be removing.
    Kept for higher-rate data from Doric and TDT rigs.
    """
    for animal_data in data.values():
        channel = animal_data[key]
        rate = 1.0 / np.median(np.diff(channel["time"])) if fs is None else fs
        if not 0 < cutoff < rate / 2:
            raise ValueError(
                f"cutoff={cutoff} Hz must be between 0 and {rate / 2} Hz for fs={rate}"
            )
        b, a = butter(order, cutoff, btype="low", fs=rate)
        channel["data"] = filtfilt(b, a, channel["data"])
    return data

def wavelet_denoise(data, key="gcamp", wavelet="db4", level=None, threshold_scale=1.0):
    """Soft-threshold wavelet detail coefficients.

    wavelet : "db4" general use, "sym4" symmetric and better for transients,
              "coif2" smoother.
    level   : decomposition depth, auto-calculated and capped at 6 if None.
    threshold_scale : higher is more aggressive.
    """
    for animal_data in data.values():
        sig = animal_data[key]["data"]
        lvl = min(pywt.dwt_max_level(len(sig), wavelet), 6) if level is None else level
        coeffs = pywt.wavedec(sig, wavelet, level=lvl)

        # MAD-based threshold, robust to outliers
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(sig)))

        # keep the approximation (slow trend), threshold the details
        out = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]

        # waverec pads, so trim back to the original length
        animal_data[key]["data"] = pywt.waverec(out, wavelet)[:len(sig)]
    return data


def savgol(data, key="gcamp", window=11, order=3):
    """Savitzky-Golay smoothing. window is in samples and must be odd and > order."""
    for animal_data in data.values():
        animal_data[key]["data"] = savgol_filter(
            animal_data[key]["data"], window_length=window, polyorder=order
        )
    return data


def median_filter(data, key="gcamp", kernel=5):
    """Median filter. Good for isolated spike artifacts, kernel must be odd."""
    for animal_data in data.values():
        animal_data[key]["data"] = medfilt(animal_data[key]["data"], kernel_size=kernel)
    return data


def wavelet_time_frequency(signal, wavelet="mexh", fs=3, scales=None):
    """Continuous wavelet transform. Returns (coefficients, frequencies).

    Takes a raw array rather than the recording dict — this is analysis, not a
    transform, so there is nothing to write back. Use it for inspecting which
    frequencies appear when, e.g. to check whether an artifact is broadband.

    wavelet : "morl" for oscillations, "mexh" for transients.
    """
    if scales is None:
        scales = np.arange(1, 128)
    return pywt.cwt(signal, scales, wavelet, sampling_period=1 / fs)




def bandpass_filter(data, key="gcamp", low=0.01, high=1.0, fs=None, order=4):
    """Butterworth bandpass — highpass and lowpass in one pass.

    Combines drift removal and fast-noise removal. Both edges must sit below
    Nyquist, which at ~3 Hz leaves very little room.
    """
    for animal_data in data.values():
        ch = animal_data[key]
        rate = 1.0 / np.median(np.diff(ch["time"])) if fs is None else fs
        if not 0 < low < high < rate / 2:
            raise ValueError(f"need 0 < low={low} < high={high} < {rate / 2} for fs={rate}")
        b, a = butter(order, [low, high], btype="band", fs=rate)
        ch["data"] = filtfilt(b, a, ch["data"])
    return data


def gaussian_smooth(data, key="gcamp", sigma=2.0):
    """Gaussian smoothing. sigma is in samples, not seconds.

    Softer than savgol with no polynomial edge artifacts, but rounds off
    transient onsets — sigma above ~3 will visibly slow your rise times.
    """
    for animal_data in data.values():
        animal_data[key]["data"] = gaussian_filter1d(animal_data[key]["data"], sigma=sigma)
    return data


def hampel_filter(data, key="gcamp", window=7, n_sigma=3.0):
    """Replace outliers with the local median, leaving everything else untouched.

    Unlike median_filter, which rewrites every sample, this only replaces
    points more than n_sigma robust deviations from the local median. Best
    choice for isolated motion spikes, since clean data passes through
    unchanged. Lower n_sigma is more aggressive.
    """
    for animal_data in data.values():
        sig = animal_data[key]["data"].copy()
        med = _medfilt_nd(sig, size=window, mode="nearest")
        mad = _medfilt_nd(np.abs(sig - med), size=window, mode="nearest")
        outliers = np.abs(sig - med) > n_sigma * 1.4826 * mad
        sig[outliers] = med[outliers]
        animal_data[key]["data"] = sig
    return data
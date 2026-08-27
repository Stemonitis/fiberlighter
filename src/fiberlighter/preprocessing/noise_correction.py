"""Denoising and smoothing.

Each function applies one method to a both channels (except wavelet trnsform is only for gcamp) of
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
-- rewrte the wavelet denoise to use both gcampt or iso
"""

import numpy as np
import pywt
from scipy.signal import savgol_filter, medfilt
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d, median_filter as _medfilt_nd
from ..registry import register_processing


@register_processing("noise_correction")

class NoiseCorrection:
    def __init__(self, recording):
        self.recording = recording

    def lowpass_filter(self, cutoff=1.0, order=4):
        """Butterworth lowpass filter applied to data. Modifies data in place.

        cutoff : cutoff frequency in Hz above which signal is attenuated. Must be below fs / 2.
        """
        if not 0 < cutoff < self.recording.fs / 2:
            raise ValueError(f"cutoff={cutoff} Hz must be between 0 and {self.recording.fs / 2} Hz for fs={self.recording.fs}")
        b, a = butter(order, cutoff, btype="low", fs=self.recording.fs)
        self.recording.iso_work = filtfilt(b, a, self.recording.iso_work)
        self.recording.gcamp_work = filtfilt(b, a, self.recording.gcamp_work)
        return self.recording

    def wavelet_denoise(self, wavelet="db4", level=None, threshold_scale=1.0):
        """Soft-threshold wavelet detail coefficients.

        wavelet : "db4" general use, "sym4" symmetric and better for transients,
                "coif2" smoother.
        level   : decomposition depth, auto-calculated and capped at 6 if None.
        threshold_scale : higher is more aggressive.
        """
        sig = self.recording.gcamp_work
        lvl = min(pywt.dwt_max_level(len(sig), wavelet), 6) if level is None else level
        coeffs = pywt.wavedec(sig, wavelet, level=lvl)

        # MAD-based threshold, robust to outliers
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(sig)))

        # keep the approximation (slow trend), threshold the details
        out = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]

        # waverec pads, so trim back to the original length
        self.recording.gcamp_work = pywt.waverec(out, wavelet)[:len(sig)]
        return self.recording

    def savgol(self, window=11, order=3):
        """Savitzky-Golay smoothing. window is in samples and must be odd and > order."""
        self.recording.gcamp_work = savgol_filter(
            self.recording.gcamp_work, window_length=window, polyorder=order
        )
        self.recording.iso_work = savgol_filter(
            self.recording.iso_work, window_length=window, polyorder=order
        )
        return self.recording

    def median_filter(self, kernel=5):
        """Median filter. Good for isolated spike artifacts, kernel must be odd."""
        self.recording.gcamp_work = medfilt(self.recording.gcamp_work, kernel_size=kernel)
        self.recording.iso_work = medfilt(self.recording.iso_work, kernel_size=kernel)
        return self.recording

    def bandpass_filter(self, low=0.01, high=1.0, fs=None, order=4):
        """Butterworth bandpass — highpass and lowpass in one pass.

        Combines drift removal and fast-noise removal. Both edges must sit below
        Nyquist, which at ~3 Hz leaves very little room.
        """
        def _run(data):
            rate = self.recording.fs
            if not 0 < low < high < rate / 2:
                raise ValueError(f"need 0 < low={low} < high={high} < {rate / 2} for fs={rate}")
            b, a = butter(order, [low, high], btype="band", fs=rate)
            return filtfilt(b, a, data)
        self.recording.iso_work = _run(self.recording.iso_work)
        self.recording.gcamp_work = _run(self.recording.gcamp_work) 
        return self.recording


    def gaussian_smooth(self, sigma=2.0):
        """Gaussian smoothing. sigma is in samples, not seconds.

        Softer than savgol with no polynomial edge artifacts, but rounds off
        transient onsets — sigma above ~3 will visibly slow your rise times.
        """
        self.recording.iso_work = gaussian_filter1d(self.recording.iso_work, sigma=sigma)
        self.recording.gcamp_work = gaussian_filter1d(self.recording.gcamp_work, sigma=sigma)
        return self.recording

    def hampel_filter(self, window=7, n_sigma=3.0):
        """Replace outliers with the local median, leaving everything else untouched.

        Unlike median_filter, which rewrites every sample, this only replaces
        points more than n_sigma robust deviations from the local median. Best
        choice for isolated motion spikes, since clean data passes through
        unchanged. Lower n_sigma is more aggressive.
        """
        def _run(data):
            sig = data.copy()
            med = _medfilt_nd(sig, size=window, mode="nearest")
            mad = _medfilt_nd(np.abs(sig - med), size=window, mode="nearest")
            outliers = np.abs(sig - med) > n_sigma * 1.4826 * mad
            sig[outliers] = med[outliers]
            return sig
        
        self.recording.iso_work = _run(self.recording.iso_work)
        self.recording.gcamp_work = _run(self.recording.gcamp_work)
        return self.recording
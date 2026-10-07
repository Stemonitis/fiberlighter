"""Normalization .

Each function applies one method to a both channels. All modify in place.

TODO / known issues
-recs on when to apply these
-right now defaults to the mean of the recording
-adding an iso fitted option potentially as a default
-having a timeframe of the baseline chunks to use instead of the mean of the whole recording
-------------------
"""
import numpy as np
# from scipy.stats import median_abs_deviation
from ..registry import register_processing

@register_processing("normalization")
class Normalization:
    def __init__(self, recording):
        self.recording = recording
    def _check_unit_type(self):
        if self.recording.unit_type != "raw":
            raise ValueError(f"Normalization can only be applied to raw data, but unit_type is {self.recording.unit_type}")
    def _check_denominator(self, value, name):
        values = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values == 0):
            raise ValueError(f"{name} must contain only finite, nonzero values")
    def _resolve_baseline(self, provided, saved, signal):
        if provided is not None:
            return provided
        if saved is not None:
            return saved
        return np.mean(signal)

    def deltaF(self, baseline_iso=None, baseline_gcamp=None):
        self._check_unit_type()
        baseline_gcamp = self._resolve_baseline(
            baseline_gcamp,
            self.recording.baseline_gcamp,
            self.recording.gcamp_work,
        )
        baseline_iso = self._resolve_baseline(
            baseline_iso,
            self.recording.baseline_iso,
            self.recording.iso_work,
        )
        self.recording.gcamp_work = self.recording.gcamp_work-baseline_gcamp
        self.recording.iso_work =  self.recording.iso_work-baseline_iso
        self.recording.unit_type = "deltaF"
        return self.recording
    def deltaF_over_Fo(self, baseline_iso=None, baseline_gcamp=None):
        self._check_unit_type()
        baseline_gcamp = self._resolve_baseline(
            baseline_gcamp,
            self.recording.baseline_gcamp,
            self.recording.gcamp_work,
        )
        baseline_iso = self._resolve_baseline(
            baseline_iso,
            self.recording.baseline_iso,
            self.recording.iso_work,
        )
        self._check_denominator(baseline_gcamp, "baseline_gcamp")
        self._check_denominator(baseline_iso, "baseline_iso")
        self.recording.gcamp_work = (self.recording.gcamp_work - baseline_gcamp) / baseline_gcamp
        self.recording.iso_work = (self.recording.iso_work - baseline_iso) / baseline_iso
        self.recording.unit_type = "df/f"
        return self.recording
    
    def deltaF_over_Fo_percent(self, baseline_iso=None, baseline_gcamp=None):
        self._check_unit_type()
        baseline_gcamp = self._resolve_baseline(
            baseline_gcamp,
            self.recording.baseline_gcamp,
            self.recording.gcamp_work,
        )
        baseline_iso = self._resolve_baseline(
            baseline_iso,
            self.recording.baseline_iso,
            self.recording.iso_work,
        )
        self._check_denominator(baseline_gcamp, "baseline_gcamp")
        self._check_denominator(baseline_iso, "baseline_iso")
        self.recording.gcamp_work = (self.recording.gcamp_work - baseline_gcamp) / baseline_gcamp * 100
        self.recording.iso_work = (self.recording.iso_work - baseline_iso) / baseline_iso * 100
        self.recording.unit_type = "df/f_percent"
        return self.recording

    def z_score(self):
        std_gcamp = np.std(self.recording.gcamp_work)
        std_iso = np.std(self.recording.iso_work)
        self._check_denominator(std_gcamp, "std_gcamp")
        self._check_denominator(std_iso, "std_iso")
        self.recording.gcamp_work = (self.recording.gcamp_work - np.mean(self.recording.gcamp_work)) / std_gcamp
        self.recording.iso_work = (self.recording.iso_work - np.mean(self.recording.iso_work)) / std_iso
        self.recording.unit_type = "z_score"
        return self.recording
    
    # def robust_z_score(self, baseline_iso=None, baseline_gcamp=None, *, scale="normal"):
    #     """Standardize both channels using their own reference median and MAD.

    #     baseline_iso, baseline_gcamp : one-dimensional array-like, optional
    #         Samples from a reference interval, in the same units and processing
    #         state as the corresponding working signal. These are reference
    #         observations, not fitted bleaching curves. If omitted, use the
    #         entire corresponding working signal. Stored bleaching baselines
    #         are not used by this method.
    #     scale : {"normal", "raw"}, default "normal"
    #         "normal" divides by approximately 1.4826 * MAD, giving a scale
    #         consistent with standard deviation for normally distributed data.
    #         "raw" divides by unscaled MAD, as in some photometry protocols.
    #         MAD = median(abs(reference - median(reference))).

    #     Inputs must be finite, nonempty, one-dimensional arrays. Each reference
    #     needs at least two samples and a positive MAD. Invalid inputs raise
    #     ValueError without changing either channel or unit_type. Returns the
    #     recording with unit_type="robust_z_score".
    #     """
    #     if scale not in ("normal", "raw"):
    #         raise ValueError("scale must be 'normal' or 'raw'")
    #     mad_scale = "normal" if scale == "normal" else 1.0

    #     def _normalize(data, baseline, channel):
    #         signal = np.asarray(data, dtype=float)
    #         if signal.ndim != 1 or signal.size == 0 or not np.all(np.isfinite(signal)):
    #             raise ValueError(f"{channel} signal must be a finite, nonempty 1D array")
    #         reference = signal if baseline is None else np.asarray(baseline, dtype=float)
    #         if reference.ndim != 1 or reference.size < 2 or not np.all(np.isfinite(reference)):
    #             raise ValueError(f"{channel} reference must be a finite 1D array with at least two samples")
    #         center = np.median(reference)
    #         spread = median_abs_deviation(reference, scale=mad_scale)
    #         if not np.isfinite(spread) or spread <= 0:
    #             raise ValueError(f"{channel} reference MAD must be positive and finite")
    #         with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
    #             normalized = (signal - center) / spread
    #         if not np.all(np.isfinite(normalized)):
    #             raise ValueError(f"{channel} robust z-score produced nonfinite values")
    #         return normalized

    #     gcamp = _normalize(self.recording.gcamp_work, baseline_gcamp, "gcamp")
    #     iso = _normalize(self.recording.iso_work, baseline_iso, "iso")
    #     self.recording.gcamp_work = gcamp
    #     self.recording.iso_work = iso
    #     self.recording.unit_type = "robust_z_score"
    #     return self.recording

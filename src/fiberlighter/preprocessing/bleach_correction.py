"""Bleaching correction methods.

Six alternative ways to remove slow drift from photometry traces.

Which to use: double_exponential is the default recommendation — bleaching
physically is a sum of a fast and a slow decay, so the fit is constrained to
shapes the process can actually produce, and it uses only real data.
single_exponential is the fallback when the double fails to converge. airpls
makes no assumption about drift shape and preserves transients well, but lam
needs tuning. polynomial and linear are included for comparison; both are
unconstrained and attenuate real signal. highpass_filter has an edge artifact.

TODO / known issues
-------------------
- The exponential fits expect RAW fluorescence. Running them on an already
  corrected signal raises "Initial guess is outside of provided bounds",
  because p0 assumes positive amplitudes and a centered signal has none.
  Do not chain two corrections.
- highpass_filter derives fs from the timestamps by default. filtfilt pads
  both ends with reflected data, so roughly one cutoff period (100 s at
  0.01 Hz) at the start and end is shaped by invented rather than measured
  samples. Bleaching is steepest at the start of a recording, exactly where
  that padding is least reliable. Inspect the edges before trusting them.
- airpls lam=1e6 is a starting point, not a validated default. It interacts
  with sampling rate and recording length. Plot the baseline over the raw
  trace for a few animals before committing to a value.
- polynomial bends to follow real transients and attenuates them (tested:
  a cubic recovered a 2.0 transient as 1.4). Order 3 is arbitrary.
- Naming is still inconsistent as a set: highpass_filter describes an
  operation, double_exponential describes a curve shape. Settle before
  anything imports these.
- Everything mutates in place. Decide mutate vs copy before JOSS.
- No provenance: a corrected recording does not record what was done to it.
- Fitted parameters are discarded. Worth returning for QC (tau values that
  differ wildly across animals usually mean a bad fit).
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend as _detrend
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ..registry import register_processing


@register_processing("bleach_correction")
class BleachCorrection:
    def __init__(self, recording):
        self.recording = recording


    def highpass_filter(self, cutoff=0.01, order=4):
        """Remove drift below cutoff frequency."""
        if not 0 < cutoff < self.recording.fs / 2:
            raise ValueError(
                f"cutoff={cutoff} Hz must be between 0 and {self.recording.fs / 2} Hz for fs={self.recording.fs}"
            )
        b, a = butter(order, cutoff, btype="high", fs=self.recording.fs)
        self.recording.iso_work = filtfilt(b, a, self.recording.iso_work)
        self.recording.gcamp_work = filtfilt(b, a, self.recording.gcamp_work)


    def double_exponential(self):
        """Fit and subtract a fast plus a slow decay. Expects raw fluorescence."""
        def _f(t, A1, tau1, A2, tau2, baseline):
            return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline
        def _run(data):
            sig = data
            p0 = [sig[0] * 0.3, 30, sig[0] * 0.7, 300, sig[-1]]
            bounds = ([0, 1, 0, 60, 0], [np.inf, 120, np.inf, 3600, np.inf])
            try:
                params, _ = curve_fit(_f, t, sig, p0=p0, bounds=bounds, maxfev=10000)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"double_exponential failed: {e}") from e
            data = sig - _f(t, *params)
            return data

        t = self.recording.time - self.recording.time[0]
        self.recording.iso_work = _run(self.recording.iso_work)
        self.recording.gcamp_work = _run(self.recording.gcamp_work)

    def single_exponential(self):
        """Fit and subtract one decay. Fallback when the double fails to converge."""
        def _f(t, A, tau, baseline):
            return A * np.exp(-t / tau) + baseline
        def _run(data):
            sig = data
            p0 = [sig[0] - sig[-1], 100, sig[-1]]
            bounds = ([0, 1, 0], [np.inf, 3600, np.inf])
            try:
                params, _ = curve_fit(_f, t, sig, p0=p0, bounds=bounds, maxfev=10000)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"single_exponential failed on: {e}") from e
            data = sig - _f(t, *params)
            return data
        t = self.recording.time - self.recording.time[0]
        self.recording.iso_work = _run(self.recording.iso_work)
        self.recording.gcamp_work = _run(self.recording.gcamp_work)


    def polynomial(self, order=3):
        """Fit and subtract a polynomial. Unconstrained — attenuates real transients."""
        t = self.recording.time - self.recording.time[0]
        sig1 = self.recording.iso_work
        sig2 = self.recording.gcamp_work

        self.recording.iso_work = sig1 - np.polyval(np.polyfit(t, sig1, order), t)
        self.recording.gcamp_work = sig2 - np.polyval(np.polyfit(t, sig2, order), t)


    def linear(self, kind="linear"):
        """Subtract a least-squares line. kind="constant" subtracts the mean only."""
        self.recording.iso_work = _detrend(self.recording.iso_work, type=kind)
        self.recording.gcamp_work = _detrend(self.recording.gcamp_work, type=kind)


    def airpls(self, lam=1e6, max_iter=15):
        """Subtract an adaptively reweighted penalised least squares baseline.

        Reweights iteratively so points above the baseline lose influence, letting
        the fit track drift without being pulled up by transients. lam sets
        stiffness; 1e5-1e8 is the usual range.
        """
        def _baseline(y):
            n = len(y)
            D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
            H = lam * D.T @ D
            w = np.ones(n)
            for i in range(1, max_iter + 1):
                z = spsolve(sparse.csc_matrix(sparse.diags(w) + H), w * y)
                d = y - z
                neg = d[d < 0]
                if len(neg) == 0 or np.abs(neg).sum() < 1e-3 * np.abs(y).sum():
                    break
                w = np.zeros(n)
                w[d < 0] = np.exp(i * np.abs(neg) / np.abs(neg).sum())
            return z
        self.recording.iso_work = self.recording.iso_work - _baseline(self.recording.iso_work)
        self.recording.gcamp_work = self.recording.gcamp_work - _baseline(self.recording.gcamp_work)
"""Bleaching correction methods.

Six alternative ways to remove slow drift from photometry traces. Each takes
the recording dict, applies one method to every channel of every animal, and
returns the same dict. All of them modify in place.

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


def _sampling_rate(time):
    """Estimate sampling rate in Hz from timestamps. Median resists dropped frames."""
    return 1.0 / np.median(np.diff(time))


def highpass_filter(data, cutoff=0.01, fs=None, order=4):
    """Remove drift below cutoff frequency. fs is derived from timestamps if None."""
    for animal_data in data.values():
        for channel in animal_data.values():
            rate = _sampling_rate(channel["time"]) if fs is None else fs
            if not 0 < cutoff < rate / 2:
                raise ValueError(
                    f"cutoff={cutoff} Hz must be between 0 and {rate / 2} Hz for fs={rate}"
                )
            b, a = butter(order, cutoff, btype="high", fs=rate)
            channel["data"] = filtfilt(b, a, channel["data"])
    return data


def double_exponential(data):
    """Fit and subtract a fast plus a slow decay. Expects raw fluorescence."""
    def _f(t, A1, tau1, A2, tau2, baseline):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline

    for name, animal_data in data.items():
        for ch_name, channel in animal_data.items():
            t = channel["time"] - channel["time"][0]
            sig = channel["data"]
            p0 = [sig[0] * 0.3, 30, sig[0] * 0.7, 300, sig[-1]]
            bounds = ([0, 1, 0, 60, 0], [np.inf, 120, np.inf, 3600, np.inf])
            try:
                params, _ = curve_fit(_f, t, sig, p0=p0, bounds=bounds, maxfev=10000)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"double_exponential failed on {name}/{ch_name}: {e}") from e
            channel["data"] = sig - _f(t, *params)
    return data


def single_exponential(data):
    """Fit and subtract one decay. Fallback when the double fails to converge."""
    def _f(t, A, tau, baseline):
        return A * np.exp(-t / tau) + baseline

    for name, animal_data in data.items():
        for ch_name, channel in animal_data.items():
            t = channel["time"] - channel["time"][0]
            sig = channel["data"]
            p0 = [sig[0] - sig[-1], 100, sig[-1]]
            bounds = ([0, 1, 0], [np.inf, 3600, np.inf])
            try:
                params, _ = curve_fit(_f, t, sig, p0=p0, bounds=bounds, maxfev=10000)
            except (RuntimeError, ValueError) as e:
                raise RuntimeError(f"single_exponential failed on {name}/{ch_name}: {e}") from e
            channel["data"] = sig - _f(t, *params)
    return data


def polynomial(data, order=3):
    """Fit and subtract a polynomial. Unconstrained — attenuates real transients."""
    for animal_data in data.values():
        for channel in animal_data.values():
            t = channel["time"] - channel["time"][0]
            sig = channel["data"]
            channel["data"] = sig - np.polyval(np.polyfit(t, sig, order), t)
    return data


def linear(data, kind="linear"):
    """Subtract a least-squares line. kind="constant" subtracts the mean only."""
    for animal_data in data.values():
        for channel in animal_data.values():
            channel["data"] = _detrend(channel["data"], type=kind)
    return data


def airpls(data, lam=1e6, max_iter=15):
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

    for animal_data in data.values():
        for channel in animal_data.values():
            channel["data"] = channel["data"] - _baseline(channel["data"])
    return data
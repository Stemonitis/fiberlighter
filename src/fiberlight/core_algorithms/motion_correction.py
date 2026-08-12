"""Motion correction: fit the iso reference to gcamp and remove it.

Each function fits iso to gcamp, subtracts the fit, and writes "dff" and
"fitted" into each animal. All modify in place.

The fit is always evaluated on ISO values, never GCaMP — evaluating on the
wrong channel is a silent, plausible-looking error.

TODO / known issues
-------------------
- normalise="mean" divides by the mean of an already-centered signal, which
  is near zero and blows up. Present for parity with the notebook; use
  "ratio" unless you know why you want it.
- ratio dF/F divides by the fitted iso, so any sample where the fit crosses
  zero produces a spike. Guard or check before trusting it.
- Expects iso already resampled onto gcamp timestamps (run interpolate first).
  No check for this; mismatched lengths will raise somewhere unhelpful.
- sliding_window_fit is O(n) polyfits and is slow on long recordings.
- sliding_window_fit attenuates transients (tested: 0.129 vs 0.170 for the
  global fit) because the window adapts to the transient itself. Window must
  be long relative to your transients.
- Whether to run this before or after bleaching correction is undecided.
- Lerner IRLS with Tukey bisquare not yet implemented.
"""

import numpy as np
from sklearn.linear_model import HuberRegressor


def _dff(sig, fitted, mode):
    if mode == "ratio":
        return (sig - fitted) / fitted
    elif mode == "mean":
        corrected = sig - fitted
        baseline = np.mean(corrected)
        return (corrected - baseline) / np.abs(baseline)
    raise ValueError(f"unknown normalise: {mode}")


def polynomial_fit(data, deg=1, normalise="ratio"):
    """Least-squares fit of iso to gcamp. deg=1 is the standard linear fit."""
    for animal_data in data.values():
        iso = animal_data["iso"]["data"]
        sig = animal_data["gcamp"]["data"]
        coeffs = np.polyfit(iso, sig, deg)
        fitted = np.polyval(coeffs, iso)   # evaluate on ISO, not GCaMP
        animal_data["fitted"] = {"time": animal_data["gcamp"]["time"], "data": fitted}
        animal_data["dff"] = {"time": animal_data["gcamp"]["time"],
                              "data": _dff(sig, fitted, normalise)}
    return data


def robust_fit(data, normalise="ratio", **kwargs):
    """Huber regression — downweights outliers instead of letting them pull the fit.

    Better than polynomial_fit when the recording has motion spikes, since
    least squares is dominated by large residuals.
    """
    for animal_data in data.values():
        iso = animal_data["iso"]["data"]
        sig = animal_data["gcamp"]["data"]
        model = HuberRegressor(**kwargs).fit(iso.reshape(-1, 1), sig)
        fitted = model.predict(iso.reshape(-1, 1))
        animal_data["fitted"] = {"time": animal_data["gcamp"]["time"], "data": fitted}
        animal_data["dff"] = {"time": animal_data["gcamp"]["time"],
                              "data": _dff(sig, fitted, normalise)}
    return data


def sliding_window_fit(data, window_sec=30, deg=1, normalise="ratio"):
    """Refit iso to gcamp in a moving window, so the relationship can drift.

    Handles recordings where the iso-gcamp coupling changes over time, at the
    cost of one polyfit per sample. Window must be long relative to your
    transients or it will fit and remove them.
    """
    for animal_data in data.values():
        iso = animal_data["iso"]["data"]
        sig = animal_data["gcamp"]["data"]
        time = animal_data["gcamp"]["time"]
        fs = 1.0 / np.median(np.diff(time))
        half = max(int(window_sec * fs) // 2, deg + 1)

        n = len(sig)
        fitted = np.zeros(n, dtype=float)
        for i in range(n):
            start, end = max(0, i - half), min(n, i + half)
            fitted[i] = np.polyval(np.polyfit(iso[start:end], sig[start:end], deg), iso[i])

        animal_data["fitted"] = {"time": time, "data": fitted}
        animal_data["dff"] = {"time": time, "data": _dff(sig, fitted, normalise)}
    return data
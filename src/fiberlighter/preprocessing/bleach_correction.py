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
from scipy.signal import butter, sosfiltfilt, detrend as _detrend
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from ..registry import register_processing


@register_processing("bleach_correction")
class BleachCorrection:
    def __init__(self, recording):
        self.recording = recording


    def highpass_filter(self, cutoff=0.0002, order=4, apply_to = "both"):
        """Remove low-frequency components below the cutoff.
        Default cuttoff is 0.01 Hz which meane the all changes slower than 100 seconds will be removed. Order  = 4 is the order of the butterworth filter. 
        add test so the cutoff is not more than the length of the recording
        it looks like at 0.0001 the bleaching is removed and gcamp has a natural curve to it, at 0.0002 its already flat at 0.00006 ther weird waves start to apper
        really hard to tell when iso and gcamp are not scaled to each other
        can we do bleach correction without scaling?
        do one bleach method on one channell and the other on the other channel design?
        maybe leave stuff in provenance instead of stuff and this way you can leave data of different formats like the different information for different methods 
        make the method return the paraameters of the fit and stuff
        how should i calculate the bleaching period maybe have an auto mode??? maybe the perfect frequency is actually bigger than the recording length
        should i change to sosfiltfilt(sos, signal) 
        implement raw vs filtered difference for diagnostics"""
        if not 0 < cutoff < self.recording.fs / 2: #check the nyqquist frequency (you can only filter up to half the sampling frequency)
            raise ValueError(
                f"cutoff={cutoff} Hz must be between 0 and {self.recording.fs / 2} Hz for fs={self.recording.fs}"
            )
        period = 1 / cutoff

        if period > self.recording.time[-1] - self.recording.time[0]:
            print(
                "Cutoff period exceeds recording duration; "
                "very-low-frequency filtering may be poorly constrained."
            )
        sos = butter(order, cutoff, btype = "high", output="sos", fs=self.recording.fs)
        if apply_to == "iso":
            self.recording.iso_work = sosfiltfilt(sos, self.recording.iso_work)
        elif apply_to == "gcamp":
            self.recording.gcamp_work = sosfiltfilt(sos, self.recording.gcamp_work)
        else:
            self.recording.iso_work = sosfiltfilt(sos, self.recording.iso_work)
            self.recording.gcamp_work = sosfiltfilt(sos, self.recording.gcamp_work)
        return self.recording


    def double_exponential(self, apply_to=both, plot_fit = False):
        """Fit and remove a fast plus a slow decay, returning ΔF/F.

        Expects raw fluorescence. The fitted curve is used as F0, so the
        baseline is time-varying rather than a single scalar.
        """
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
            curve = _f(t, *params)
            if plot_fit:
                plt.figure()
                plt.plot(t, sig, label="raw")
                plt.plot(t, curve, label="fit")
                plt.legend()
                plt.title("Double exponential fit")
                plt.xlabel("Time (s)")
                plt.ylabel("Fluorescence")
                plt.show()
            return curve

        t = self.recording.time - self.recording.time[0]
        self.baseline_iso = _run(self.recording.iso_work)
        self.baseline_gcamp = _run(self.recording.gcamp_work)
        self.recording.iso_work = self.recording.iso_work - self.baseline_iso
        self.recording.gcamp_work = self.recording.gcamp_work - self.baseline_gcamp
        return self.recording


 def double_exponential(self, plot_fit=False, convert_to_deltaF_over_Fo=True,
#                            apply_to='both', tau_fast_bounds=(1, 120),
#                            tau_slow_bounds=(60, 3600)):
#         """Fit and remove two exponential decays plus a constant baseline.

#         Time constants and bounds are in seconds, with the original bounds
#         restored by default. Adjust them for your recording; they are not
#         universal bleaching times. A small component can still make the sum
#         look like one exponential. Set convert_to_deltaF_over_Fo=False to
#         subtract only. plot_fit creates a figure; call plt.show() to display.
#         """
#         fast = np.asarray(tau_fast_bounds, dtype=float)
#         slow = np.asarray(tau_slow_bounds, dtype=float)
#         for bounds in (fast, slow):
#             if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
#                 raise ValueError('Time-constant bounds must be positive (lower, upper) pairs')

#         def _f(t, A1, tau1, A2, tau2, baseline):
#             return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline

#         def _run(t, sig):
#             if len(sig) < 6 or np.any(sig < 0) or np.max(sig) <= 0:
#                 raise ValueError('Double exponential needs at least six nonnegative fluorescence samples')
#             # Estimate the decay amplitude, not the entire fluorescence level:
#             # the constant baseline already accounts for most of that level.
#             n = max(1, len(sig) // 100)
#             baseline = max(float(np.median(sig[-n:])), 0.)
#             amplitude = max(float(np.median(sig[:n])) - baseline, .01 * np.max(sig))
#             tau1 = float(np.clip(30., fast[0], fast[1]))
#             tau2 = float(np.clip(300., slow[0], slow[1]))
#             p0 = [amplitude * .3, tau1, amplitude * .7, tau2, baseline]
#             bounds = ([0, fast[0], 0, slow[0], 0],
#                       [np.inf, fast[1], np.inf, slow[1], np.inf])
#             try:
#                 params, covariance = curve_fit(_f, t, sig, p0=p0, bounds=bounds,
#                                                maxfev=20000, x_scale='jac')
#             except (RuntimeError, ValueError) as exc:
#                 raise RuntimeError(f'double_exponential failed: {exc}') from exc
#             A1, tau1, A2, tau2, baseline = params
#             return _f(t, *params), dict(
#                 parameters=dict(amplitude_fast=A1, tau_fast=tau1,
#                                 amplitude_slow=A2, tau_slow=tau2, offset=baseline),
#                 covariance=covariance,
#                 components={'fast decay': A1 * np.exp(-t / tau1),
#                             'slow decay': A2 * np.exp(-t / tau2)},
#                 tau_bounds={'fast': tuple(fast), 'slow': tuple(slow)},
#             )

#         return self._fit('double_exponential', _run, apply_to, plot_fit,
#                          convert_to_deltaF_over_Fo)






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
        return self.recording


    def polynomial(self, order=3):
        """Fit and subtract a polynomial. Unconstrained — attenuates real transients."""
        t = self.recording.time - self.recording.time[0]
        sig1 = self.recording.iso_work
        sig2 = self.recording.gcamp_work

        self.recording.iso_work = sig1 - np.polyval(np.polyfit(t, sig1, order), t)
        self.recording.gcamp_work = sig2 - np.polyval(np.polyfit(t, sig2, order), t)
        return self.recording


    def linear(self, kind="linear"):
        """Subtract a least-squares line. kind="constant" subtracts the mean only."""
        self.recording.iso_work = _detrend(self.recording.iso_work, type=kind)
        self.recording.gcamp_work = _detrend(self.recording.gcamp_work, type=kind)
        return self.recording


    def airpls(self, lam=1e9, max_iter=15):
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
        return self.recording





#         """In-place bleaching corrections with saved fits and explicit channel selection.

# Fits describe slow trends, not uniquely identified photobleaching. They may
# remove slow biological responses. Inspect saved fits before interpreting data.
# All selected channels are computed and validated before any are changed.
# """
# import warnings
# import numpy as np
# from scipy.signal import butter, sosfiltfilt
# from scipy.optimize import curve_fit
# from scipy import sparse
# from scipy.sparse.linalg import spsolve
# from ..registry import register_processing


# @register_processing("bleach_correction")
# class BleachCorrection:
#     def __init__(self, recording):
#         self.recording = recording

#     def _inputs(self, apply_to, raw=False):
#         if apply_to not in ('both', 'iso', 'gcamp'):
#             raise ValueError("apply_to must be 'both', 'iso' or 'gcamp'")
#         rec = self.recording
#         t = np.asarray(rec.time, dtype=float)
#         if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
#             raise ValueError('Need at least four finite, increasing timestamps')
#         channels = ('iso', 'gcamp') if apply_to == 'both' else (apply_to,)
#         signals = {}
#         for ch in channels:
#             if raw and rec.channel_units[ch] != 'raw':
#                 raise ValueError(f'{ch} must be raw before fitting a bleaching baseline; reset first')
#             y = np.asarray(getattr(rec, ch + '_work'), dtype=float)
#             if y.shape != t.shape or not np.isfinite(y).all():
#                 raise ValueError(f'{ch} must be finite and match time')
#             signals[ch] = y.copy()
#         return t - t[0], signals

#     def _fit(self, method, estimator, apply_to, plot_fit, convert):
#         rec = self.recording
#         t, signals = self._inputs(apply_to, raw=True)
#         pending = {}
#         for ch, y in signals.items():
#             baseline, details = estimator(t, y)
#             if baseline.shape != y.shape or not np.isfinite(baseline).all():
#                 raise ValueError(f'{ch}: fitted baseline is invalid')
#             if convert and np.any(baseline <= np.finfo(float).eps * max(1., np.max(abs(y)))):
#                 raise ValueError(f'{ch}: ΔF/F needs a positive baseline safely above zero')
#             corrected = y - baseline
#             if convert:
#                 corrected = corrected / baseline
#             if not np.isfinite(corrected).all():
#                 raise ValueError(f'{ch}: correction produced nonfinite values')
#             pending[ch] = (corrected, dict(method=method, input=y, baseline=baseline,
#                                            time=np.asarray(rec.time).copy(),
#                                            output_unit='df/f' if convert else 'deltaF', **details))
#         for ch, (corrected, fit) in pending.items():
#             setattr(rec, ch + '_work', corrected)
#             setattr(rec, 'baseline_' + ch, fit['baseline'].copy())
#             rec.bleach_fits[ch] = fit
#             rec.channel_units[ch] = fit['output_unit']
#         rec._sync_units()
#         if plot_fit:
#             rec.visualization.plot_raw_with_baseline(apply_to=apply_to)
#         return rec

#     def double_exponential(self, plot_fit=False, convert_to_deltaF_over_Fo=True,
#                            apply_to='both', tau_fast_bounds=(1, 120),
#                            tau_slow_bounds=(60, 3600)):
#         """Fit and remove two exponential decays plus a constant baseline.

#         Time constants and bounds are in seconds, with the original bounds
#         restored by default. Adjust them for your recording; they are not
#         universal bleaching times. A small component can still make the sum
#         look like one exponential. Set convert_to_deltaF_over_Fo=False to
#         subtract only. plot_fit creates a figure; call plt.show() to display.
#         """
#         fast = np.asarray(tau_fast_bounds, dtype=float)
#         slow = np.asarray(tau_slow_bounds, dtype=float)
#         for bounds in (fast, slow):
#             if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
#                 raise ValueError('Time-constant bounds must be positive (lower, upper) pairs')

#         def _f(t, A1, tau1, A2, tau2, baseline):
#             return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline

#         def _run(t, sig):
#             if len(sig) < 6 or np.any(sig < 0) or np.max(sig) <= 0:
#                 raise ValueError('Double exponential needs at least six nonnegative fluorescence samples')
#             # Estimate the decay amplitude, not the entire fluorescence level:
#             # the constant baseline already accounts for most of that level.
#             n = max(1, len(sig) // 100)
#             baseline = max(float(np.median(sig[-n:])), 0.)
#             amplitude = max(float(np.median(sig[:n])) - baseline, .01 * np.max(sig))
#             tau1 = float(np.clip(30., fast[0], fast[1]))
#             tau2 = float(np.clip(300., slow[0], slow[1]))
#             p0 = [amplitude * .3, tau1, amplitude * .7, tau2, baseline]
#             bounds = ([0, fast[0], 0, slow[0], 0],
#                       [np.inf, fast[1], np.inf, slow[1], np.inf])
#             try:
#                 params, covariance = curve_fit(_f, t, sig, p0=p0, bounds=bounds,
#                                                maxfev=20000, x_scale='jac')
#             except (RuntimeError, ValueError) as exc:
#                 raise RuntimeError(f'double_exponential failed: {exc}') from exc
#             A1, tau1, A2, tau2, baseline = params
#             return _f(t, *params), dict(
#                 parameters=dict(amplitude_fast=A1, tau_fast=tau1,
#                                 amplitude_slow=A2, tau_slow=tau2, offset=baseline),
#                 covariance=covariance,
#                 components={'fast decay': A1 * np.exp(-t / tau1),
#                             'slow decay': A2 * np.exp(-t / tau2)},
#                 tau_bounds={'fast': tuple(fast), 'slow': tuple(slow)},
#             )

#         return self._fit('double_exponential', _run, apply_to, plot_fit,
#                          convert_to_deltaF_over_Fo)

#     def single_exponential(self, plot_fit=False, convert_to_deltaF_over_Fo=False,
#                            apply_to='both', tau_bounds=(1, 3600)):
#         """Fit and subtract A exp(-t/tau)+baseline, optionally converting to ΔF/F."""
#         limits = np.asarray(tau_bounds, dtype=float)
#         if limits.shape != (2,) or not np.isfinite(limits).all() or not 0 < limits[0] < limits[1]:
#             raise ValueError('tau_bounds must be a positive (lower, upper) pair')

#         def _f(t, A, tau, baseline):
#             return A * np.exp(-t / tau) + baseline

#         def _run(t, sig):
#             if np.any(sig < 0) or np.max(sig) <= 0:
#                 raise ValueError('Single exponential requires nonnegative fluorescence')
#             n = max(1, len(sig) // 100)
#             baseline = max(float(np.median(sig[-n:])), 0.)
#             amplitude = max(float(np.median(sig[:n])) - baseline, .01 * np.max(sig))
#             p0 = [amplitude, float(np.clip(100., *limits)), baseline]
#             try:
#                 params, covariance = curve_fit(
#                     _f, t, sig, p0=p0, bounds=([0, limits[0], 0], [np.inf, limits[1], np.inf]),
#                     maxfev=20000, x_scale='jac')
#             except (RuntimeError, ValueError) as exc:
#                 raise RuntimeError(f'single_exponential failed: {exc}') from exc
#             A, tau, baseline = params
#             return _f(t, *params), dict(
#                 parameters=dict(amplitude=A, tau=tau, offset=baseline),
#                 covariance=covariance, components={'decay': A * np.exp(-t / tau)})

#         return self._fit('single_exponential', _run, apply_to, plot_fit,
#                          convert_to_deltaF_over_Fo)

#     def polynomial(self, order=3, plot_fit=False, convert_to_deltaF_over_Fo=False, apply_to='both'):
#         """Fit a polynomial on scaled time. High orders can remove biological changes."""
#         if isinstance(order,bool) or not isinstance(order,(int,np.integer)) or order < 0:
#             raise ValueError('order must be a nonnegative integer')
#         def estimate(t,y):
#             if order >= len(t):
#                 raise ValueError('order must be less than sample count')
#             fit = np.polynomial.Polynomial.fit(t,y,order)
#             return fit(t), dict(parameters={'coefficients':fit.convert().coef, 'order':order})
#         return self._fit('polynomial', estimate, apply_to, plot_fit, convert_to_deltaF_over_Fo)

#     def linear(self, kind='linear', plot_fit=False, convert_to_deltaF_over_Fo=False, apply_to='both'):
#         """Subtract a time-based least-squares line or a constant mean baseline."""
#         if kind not in ('linear','constant'):
#             raise ValueError("kind must be 'linear' or 'constant'")
#         def estimate(t,y):
#             fit = np.polynomial.Polynomial.fit(t,y,1 if kind == 'linear' else 0)
#             return fit(t), dict(parameters={'coefficients':fit.convert().coef, 'kind':kind})
#         return self._fit('linear', estimate, apply_to, plot_fit, convert_to_deltaF_over_Fo)

#     def airpls(self, lam=1e9, max_iter=15, plot_fit=False, convert_to_deltaF_over_Fo=False, apply_to='both'):
#         """Adaptive penalized baseline; lam depends on sample rate and trace length.

#         This estimate is not necessarily a physiological fluorescence baseline.
#         ΔF/F conversion is optional and requires a positive fitted baseline.
#         """
#         if not np.isfinite(lam) or lam <= 0 or isinstance(max_iter,bool) or not isinstance(max_iter,(int,np.integer)) or max_iter < 1:
#             raise ValueError('lam must be positive and max_iter a positive integer')
#         def estimate(t,y):
#             n = len(y)
#             D = sparse.diags([1.,-2.,1.],[0,1,2],shape=(n-2,n),format='csc')
#             H = lam*(D.T@D)
#             w = np.ones(n)
#             converged = False
#             for i in range(1,max_iter+1):
#                 z = spsolve(sparse.diags(w,format='csc')+H,w*y)
#                 d = y-z
#                 neg = d[d<0]
#                 total = np.abs(neg).sum()
#                 if total <= 1e-3*max(np.abs(y).sum(),np.finfo(float).eps):
#                     converged = True
#                     break
#                 w = np.zeros(n)
#                 w[d<0] = np.exp(np.minimum(i*np.abs(neg)/total,50))
#                 w[[0,-1]] = np.exp(min(i*np.max(np.abs(neg))/total,50))
#             if not converged:
#                 warnings.warn('airPLS reached max_iter; inspect the saved baseline', RuntimeWarning)
#             return z, dict(parameters={'lam':lam,'max_iter':max_iter}, iterations=i, converged=converged)
#         return self._fit('airpls', estimate, apply_to, plot_fit, convert_to_deltaF_over_Fo)



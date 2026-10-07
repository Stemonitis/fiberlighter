"""Exploratory occurrence plots; these methods have different denominators.

References: Cole & Voytek (2019), doi:10.1152/jn.00273.2019;
BOSC overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC9828710/;
https://emd.readthedocs.io/en/stable/emd_tutorials/02_spectrum_analysis/emd_tutorial_02_spectrum_01_hilberthuang.html
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, fftconvolve
from scipy.stats import chi2


def compare_oscillations(rec, channel="gcamp", period_seconds=(10, 300),
                         time_bin_seconds=300, bins=20, min_cycles=3,
                         power_percentile=0.95, cycle_thresholds=None, method=None):
    """Return (fig, axes, results) for three occurrence methods on one raw channel.

    Rows: bycycle burst-cycle counts; basic BOSC-style percent time detected;
    unweighted EMD frequency occupancy (component-seconds, summed over IMFs).
    Time heatmaps sit below the raw recording with a shared time axis.
    Whole-recording frequency histograms sit to their right. Returned axes
    remain [raw, histogram, heatmap, ...].
    Linear detrending is applied to a copy for all methods. A common edge
    exclusion of two longest periods is used. No recording arrays change.

    BOSC-style uses unit-energy, six-cycle Morlet wavelets, a straight-line
    log(mean wavelet power) versus log(frequency) background fit, chi-square
    power threshold and a minimum-cycle duration. It is NOT eBOSC/fBOSC:
    no peak removal, knee model or episode de-duplication is performed.
    Occupancy across frequencies must not be summed. Background contamination
    by oscillations or bleaching can bias detection. Thresholds are exploratory.

    EMD uses mask_sift (up to six IMFs) and the normalized Hilbert transform.
    Every finite in-range IMF sample contributes 1/fs seconds, irrespective
    of amplitude. Overlapping IMFs contribute separately; this is neither
    cycle count nor percent recording time. Noise modes also contribute.
    """
    if method not in (None, "bycycle", "bosc", "emd"):
        raise ValueError("Unknown occurrence method")
    if method in (None, "bycycle"):
        from bycycle.features import compute_shape_features, compute_burst_features
        from bycycle.burst.utils import check_min_burst_cycles
        import pandas as pd
    if method in (None, "emd"):
        import emd
    if channel not in ("gcamp", "iso"):
        raise ValueError("channel must be 'gcamp' or 'iso'")
    t = np.asarray(rec.time, dtype=float)
    x = np.asarray(getattr(rec, channel), dtype=float)
    p = np.asarray(period_seconds, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all() or not 0 < p[0] < p[1]:
        raise ValueError("period_seconds must be (shortest, longest), positive and ordered")
    if t.ndim != 1 or len(t) < 8 or x.shape != t.shape or not (np.isfinite(t).all() and np.isfinite(x).all()):
        raise ValueError("Signal and timestamps must be matching finite arrays")
    fs = rec.fs
    if not np.isfinite(fs) or fs <= 0 or not np.allclose(np.diff(t), 1/fs, rtol=.01, atol=0):
        raise ValueError("Evenly sampled timestamps consistent with fs are required")
    if 1/p[0] >= fs/2 or np.ptp(x) == 0:
        raise ValueError("Need a nonconstant signal and frequencies below Nyquist")
    if not isinstance(bins, int) or bins < 3 or not np.isfinite(time_bin_seconds) or time_bin_seconds <= 0:
        raise ValueError("Use at least three frequency bins and a positive time bin")
    if not np.isfinite(min_cycles) or min_cycles < 1 or not 0 < power_percentile < 1:
        raise ValueError("min_cycles must be >=1 and power_percentile between 0 and 1")
    t = t-t[0]
    valid = (t >= 2*p[1]) & (t <= t[-1]-2*p[1])
    if valid.sum()/fs < min_cycles*p[1]:
        raise ValueError("Recording too short: allow four longest periods for edges plus min_cycles periods")
    sig = detrend(x)
    edges = np.geomspace(1/p[1], 1/p[0], bins+1)
    frequencies = np.sqrt(edges[:-1]*edges[1:])
    te = np.arange(0, t[-1], time_bin_seconds)
    te = np.r_[te, t[-1]]
    if method in (None, "bycycle"):
        thresholds = dict(amp_fraction_threshold=.2, amp_consistency_threshold=.5,
                          period_consistency_threshold=.5, monotonicity_threshold=.8,
                          min_n_cycles=int(np.ceil(min_cycles)))
        if cycle_thresholds is not None:
            thresholds.update(cycle_thresholds)
        # neurodsp checks filters on int(fs*2) response points. Rescale time
        # units for sub-Hz data so that check has adequate resolution. Ratios
        # f/fs, filter coefficients and sample-based cycle periods are unchanged.
        unit_scale = max(1., 4/edges[0])
        shapes = compute_shape_features(sig, fs*unit_scale,
                                        (edges[0]*unit_scale, edges[-1]*unit_scale))
        features = compute_burst_features(shapes, sig)
        cycles = pd.concat([features, shapes], axis=1)
        # Equivalent bycycle consistency criteria, with a writable NumPy mask
        # for pandas 3 (bycycle 1.2's combined helper writes to a read-only view).
        accepted = np.ones(len(cycles), dtype=bool)
        for name in ('amp_fraction', 'amp_consistency', 'period_consistency', 'monotonicity'):
            threshold_value = thresholds[name + '_threshold']
            if not 0 <= threshold_value <= 1:
                raise ValueError('Cycle consistency thresholds must be between 0 and 1')
            accepted &= cycles[name].to_numpy() > threshold_value
        if len(accepted):
            accepted[[0, -1]] = False
        cycles['is_burst'] = check_min_burst_cycles(accepted, min_n_cycles=thresholds['min_n_cycles'])
        cf = fs / cycles['period'].to_numpy()
        ct = cycles['sample_peak'].to_numpy()/fs
        keep = (cycles['is_burst'].to_numpy() &
                (cycles['sample_last_trough'].to_numpy()/fs >= 2*p[1]) &
                (cycles['sample_next_trough'].to_numpy()/fs <= t[-1]-2*p[1]))
        cycle_map = np.histogram2d(cf[keep], ct[keep], bins=(edges, te))[0]

    if method in (None, "bosc"):
        # Basic BOSC-style detection: retain runs exceeding both thresholds.
        power = np.empty((bins, len(sig)))
        for i, f in enumerate(frequencies):
            sigma = 6/(2*np.pi*f)
            wt = np.arange(-int(np.ceil(3.5*sigma*fs)), int(np.ceil(3.5*sigma*fs))+1)/fs
            wave = np.exp(2j*np.pi*f*wt)*np.exp(-wt**2/(2*sigma**2))
            wave /= np.sqrt(np.sum(abs(wave)**2))
            power[i] = abs(fftconvolve(sig, wave, mode='same'))**2
        background_fit = np.polyfit(np.log10(frequencies), np.log10(power[:, valid].mean(axis=1)), 1)
        background = 10**np.polyval(background_fit, np.log10(frequencies))
        threshold = background*chi2.ppf(power_percentile, 2)/2
        detected = np.zeros(power.shape, dtype=bool)
        for i, f in enumerate(frequencies):
            mask = (power[i] > threshold[i]) & valid
            transitions = np.diff(np.r_[False, mask, False].astype(int))
            for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                if end-start >= np.ceil(min_cycles*fs/f):
                    detected[i, start:end] = True
        occupancy = np.full((bins, len(te)-1), np.nan)
        for j in range(len(te)-1):
            within = valid & (t >= te[j]) & ((t < te[j+1]) if j < len(te)-2 else (t <= te[j+1]))
            if within.any():
                occupancy[:, j] = 100*detected[:, within].mean(axis=1)

    if method in (None, "emd"):
        imfs = emd.sift.mask_sift(sig, max_imfs=6)
        _, instantaneous_frequency, _ = emd.spectra.frequency_transform(imfs, fs, 'nht')
        emd_map = np.zeros((bins, len(te)-1))
        for f in instantaneous_frequency.T:
            ok = valid & np.isfinite(f) & (f >= edges[0]) & (f <= edges[-1])
            emd_map += np.histogram2d(f[ok], t[ok], bins=(edges, te))[0]/fs
    empty = np.histogram(t[valid], te)[0] == 0
    entries = []
    results = dict(frequency_edges=edges, time_edges_seconds=te,
                   detrended_signal=sig, valid_samples=valid,
                   parameters=dict(channel=channel, period_seconds=tuple(p), min_cycles=min_cycles,
                                   power_percentile=power_percentile, time_bin_seconds=time_bin_seconds,
                                   method=method))
    if method in (None, "bycycle"):
        cycle_map[:, empty] = np.nan
        entries.append((cycle_map, np.nansum(cycle_map, axis=1), 'bycycle · accepted cycles', None))
        results.update(cycle_counts=cycle_map, cycle_features=cycles, cycle_thresholds=thresholds)
    if method in (None, "bosc"):
        entries.append((occupancy, 100*detected[:, valid].mean(axis=1),
                        'Basic BOSC-style · time detected (%)', 100))
        results.update(bosc_occupancy_percent=occupancy, background_power=background, power_threshold=threshold)
    if method in (None, "emd"):
        emd_map[:, empty] = np.nan
        entries.append((emd_map, np.nansum(emd_map, axis=1), 'Unweighted EMD · component-seconds', None))
        results.update(emd_component_seconds=emd_map, imfs=imfs)
    results['whole_recording'] = [entry[1] for entry in entries]
    fig = plt.figure(figsize=(14, 4+2.7*len(entries)), layout='constrained')
    grid = fig.add_gridspec(len(entries)+1, 3, width_ratios=[4, .12, 2],
                           height_ratios=[.8]+[1]*len(entries))
    raw = fig.add_subplot(grid[0, 0])
    raw.plot(t/60, x, lw=.7, color='#7556c7' if channel == 'gcamp' else '#168c87',
             label=f'Raw {channel.upper()} (recording.{channel})')
    raw.legend(frameon=False)
    raw.set(xlabel='Elapsed time (minutes)', ylabel='Raw intensity', title=channel.upper())
    raw.set_xlim(0, t[-1]/60)
    axes = [raw]
    cmap = plt.get_cmap('turbo').with_extremes(bad='#d9d9d9')
    for row, (matrix, total, label, maximum) in enumerate(entries, start=1):
        left = fig.add_subplot(grid[row, 2])
        right = fig.add_subplot(grid[row, 0], sharex=raw)
        left.stairs(total, edges, fill=True, color='#1976d2', alpha=.8)
        left.set(xscale='log', xlabel='Frequency (Hz)', ylabel=label, xlim=(edges[0], edges[-1]),
                 ylim=(0, max(1, float(np.max(total))*1.1)))
        left.set_title('Whole-recording frequency summary', fontsize=10)
        if label.startswith('bycycle') and total.sum() == 0:
            left.text(.5, .5, 'No cycles passed the selected burst criteria',
                      transform=left.transAxes, ha='center', fontsize=10)
        mesh = right.pcolormesh(te/60, edges, np.ma.masked_invalid(matrix), cmap=cmap,
                               vmin=0, vmax=maximum if maximum is not None else max(1, np.nanmax(matrix)), rasterized=True)
        right.set(yscale='log', xlabel='Elapsed time (minutes)', ylabel='Frequency (Hz)')
        fig.colorbar(mesh, cax=fig.add_subplot(grid[row, 1]), label=label)
        axes.extend([left, right])
    fig.suptitle(('Oscillation occurrence comparison' if method is None else entries[0][2]) + '\n'
                 'Linear detrending; gray = excluded edges. Exploratory, not validated photometry event detection.')
    return fig, axes, results

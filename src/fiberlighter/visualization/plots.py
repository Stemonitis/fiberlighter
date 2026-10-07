import pywt
import numpy as np
import matplotlib.pyplot as plt
from ..registry import register_processing


@register_processing("visualization")

class Visualization:
    def __init__(self, recording):
        self.recording = recording
    @staticmethod
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
    def basic_plot(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.recording.time, self.recording.gcamp_work, label="GCaMP")
        ax.plot(self.recording.time, self.recording.iso_work, label="Iso")
        for event_name, event_time in self.recording.events.items():
            for index, value in enumerate(np.atleast_1d(event_time)):
                ax.axvline(value, linestyle="--", label=str(event_name) if index == 0 else None)
        ax.set_xlabel("Time (s)")
        units = self.recording.channel_units
        ax.set_ylabel(" / ".join(f"{ch}: {units[ch]}" for ch in ("gcamp", "iso")))
        ax.legend()
        return self.recording
    def plot_gcamp(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.recording.time, self.recording.gcamp_work, label="GCaMP")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.recording.channel_units["gcamp"])
        return self.recording
    
    def plot_raw_with_baseline(self, ax=None, apply_to="gcamp", show_components=False):
        """Plot the saved fit input, baseline and optional exponential components.

        Returns the recording for chaining; does not call plt.show(). For both
        channels, supply two axes or let this method create them. Saved inputs
        are used even after correction/normalization changes the working data.
        """
        if apply_to not in ("gcamp", "iso", "both"):
            raise ValueError("apply_to must be 'gcamp', 'iso' or 'both'")
        channels = ("iso", "gcamp") if apply_to == "both" else (apply_to,)
        fits = []
        for ch in channels:
            fit = self.recording.bleach_fits.get(ch)
            if fit is None:
                raise ValueError(f"No saved bleaching fit for {ch}")
            fits.append(fit)
        if ax is None:
            _, axes = plt.subplots(len(channels), 1, squeeze=False, figsize=(11,4*len(channels)), layout="constrained")
            axes = axes.ravel()
        else:
            axes = np.asarray(ax, dtype=object).reshape(-1)
            if len(axes) != len(channels):
                raise ValueError("Provide one axis per selected channel")
        for axis, ch, fit in zip(axes,channels,fits):
            t = fit['time']
            axis.plot(t, fit['input'], lw=.8, alpha=.7, label=f"{ch}: fit input")
            axis.plot(t, fit['baseline'], color='black', lw=2, label='Fitted baseline')
            params = fit['parameters']
            if show_components:
                for name, component in fit.get('components', {}).items():
                    axis.plot(t, component+params.get('offset',0), '--', lw=1,
                              label=name+' + offset')
            taus = ', '.join(f'{name}={value:.3g} s' for name,value in params.items() if name.startswith('tau'))
            axis.set(title=f"{ch.upper()} · {fit['method']}" + (f" · {taus}" if taus else ''),
                     xlabel='Time (s)', ylabel='Fluorescence (source units)')
            axis.legend()
        return self.recording

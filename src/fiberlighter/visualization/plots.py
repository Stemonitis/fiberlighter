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
    def plot_with_fitted_iso(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.recording.time, self.recording.gcamp_work, label="GCaMP")
        ax.plot(self.recording.time, self.recording.iso_work, label="Iso")
        ax.plot(self.recording.time, self.recording.iso_fitted, label="Iso_fitted")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.recording.channel_units["gcamp"])
        return self.recording
    def plot_gcamp(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.recording.time, self.recording.gcamp_work, label="GCaMP")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(self.recording.channel_units["gcamp"])
        return self.recording
    
    def plot_raw_with_baseline(self, ax=None):
        """Plot raw GCaMP above raw ISO, each with its fitted baseline.

        Fit baselines with bleach_correction.double_exponential() first.
        Accepts two axes and returns the recording for chaining. Call plt.show()
        to display the figure, as with the other plotting methods.
        """
        correction = self.recording.bleach_correction
        baseline_gcamp = getattr(correction, "baseline_gcamp", None)
        baseline_iso = getattr(correction, "baseline_iso", None)
        if baseline_gcamp is None or baseline_iso is None:
            raise ValueError("Fit both bleaching baselines before plotting")

        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 8), layout="constrained")
        ax = np.asarray(ax, dtype=object).reshape(-1)
        if len(ax) != 2:
            raise ValueError("Provide two axes: GCaMP on top and ISO on the bottom")

        ax[0].plot(self.recording.time, self.recording.gcamp, label="Raw GCaMP")
        ax[0].plot(self.recording.time, baseline_gcamp, color="black", label="Fitted baseline")
        ax[0].set_title("GCaMP")
        ax[1].plot(self.recording.time, self.recording.iso, label="Raw ISO")
        ax[1].plot(self.recording.time, baseline_iso, color="black", label="Fitted baseline")
        ax[1].set_title("ISO")
        for axis in ax:
            axis.set_ylabel("Fluorescence")
            axis.legend()
        ax[1].set_xlabel("Time (s)")
        return self.recording

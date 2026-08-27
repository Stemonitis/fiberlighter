import pywt
import numpy as np
import matplotlib.pyplot as plt
from ..registry import register_processing


@register_processing("visualization")

class Visualization:
    def __init__(self, recording):
        self.recording = recording
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

        ax.plot(self.recording.time, self.recording.gcamp_work)
        ax.plot(self.recording.time, self.recording.iso_work)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dF/F")
        return self.recording
    def plot_gcamp(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.recording.time, self.recording.gcamp_work)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("dF/F")
        return self.recording

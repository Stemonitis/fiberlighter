import numpy as np
import copy
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff=1.0, fs=3, order=4):
    """Butterworth lowpass filter applied to every channel. Modifies data in place.

    cutoff : cutoff frequency in Hz above which signal is attenuated. Must be below fs / 2.
    fs     : sampling rate in Hz.
    """
    if not 0 < cutoff < fs / 2:
        raise ValueError(f"cutoff={cutoff} Hz must be between 0 and {fs / 2} Hz for fs={fs}")
    b, a = butter(order, cutoff, btype="low", fs=fs)
    for animal_data in data.values():
        for channel in animal_data.values():
            channel["data"] = filtfilt(b, a, channel["data"])
    return data
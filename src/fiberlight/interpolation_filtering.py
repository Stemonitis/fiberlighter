import numpy as np
import copy
from scipy.signal import butter, filtfilt


def interpolate(data):
    data_copy = copy.deepcopy(data)
    for animal_key, animal_data in data.items():
        data_copy[animal_key]["iso_interp"] = np.interp(animal_data["gcamp"]["time"], animal_data["iso"]["time"], animal_data["iso"]["data"])
    return data_copy
        

def lowpass_filter(signal, cutoff, fs=10, order=4):
    """Butterworth lowpass filter.

    cutoff : frequency in Hz above which signal is attenuated.
             Must be below fs / 2.
    fs     : sampling rate in Hz.
    """
    if not 0 < cutoff < fs / 2:
        raise ValueError(f"cutoff={cutoff} Hz must be between 0 and {fs / 2} Hz for fs={fs}")
    b, a = butter(order, cutoff, btype="low", fs=fs)
    return filtfilt(b, a, signal)
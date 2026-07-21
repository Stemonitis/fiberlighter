def interpolate(data_gcamp, data_iso, time_gcamp, time_iso):
        interp_func = interp1d(time_iso, data_iso, kind='linear', fill_value='extrapolate')
        iso_aligned = interp_func(time_gcamp)
        if (len(data_gcamp.shape)==2):
            # Multiple animals
            return [interpolate(d) for d in data]
        else:
            # Single animal
            return interpolate(data)


def lowpass_filter(signal, cutoff=10, fs=3, order=4):
    """Lowpass filter to remove fast noise"""
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def wavelet_denoise(signal, wavelet='db4', level=None, threshold_scale=1.0):
    """
    Denoise signal using wavelet transform
    
    Parameters
    ----------
    signal : array
        Input signal
    wavelet : str
        Wavelet type. Good options:
        - 'db4' (Daubechies 4) - good for general use
        - 'sym4' (Symlet 4) - symmetric, good for transients
        - 'coif2' (Coiflet 2) - smooth
    level : int, optional
        Decomposition level. If None, auto-calculate
    threshold_scale : float
        Multiplier for threshold (higher = more aggressive denoising)
        
    Returns
    -------
    denoised : array
        Denoised signal
    """
    
    # Auto-calculate decomposition level if not provided
    if level is None:
        level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(level, 6)  # Cap at 6 for speed
    
    # Decompose signal
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate threshold using MAD (Median Absolute Deviation)
    # This is robust to outliers
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Threshold detail coefficients (keep approximation untouched)
    coeffs_thresh = [coeffs[0]]  # Keep approximation (slow trend)
    
    for i in range(1, len(coeffs)):
        # Soft thresholding
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    
    # Reconstruct signal
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    
    # Handle length mismatch due to padding
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    
    return denoised



def wavelet_time_frequency(signal, time, wavelet='mexh', fs=5):
    """
    Continuous wavelet transform - see signal at multiple scales
    
    This creates a "spectrogram" showing which frequencies
    are present at which times
    
    Parameters
    ----------
    signal : array
    time : array
    wavelet : str
        'morl' (Morlet) - good for oscillations
        'mexh' (Mexican hat) - good for transients
    fs : float
        Sampling frequency
    
    Returns
    -------
    coefficients : 2D array (scales × time)
    frequencies : array of frequencies
    """
    
    # Define scales (convert to frequencies)
    scales = np.arange(1, 128)
    
    # Perform continuous wavelet transform
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    
    return coefficients, frequencies

def savgol(signal, method='lowpass', **kwargs):
   return savgol_filter(signal, window_length=kwargs.get('window', 11), polyorder=kwargs.get('order', 3))

def median_filtering(signal, method='lowpass', **kwargs):
    return medfilt(signal, kernel_size=kwargs.get('kernel', 5))


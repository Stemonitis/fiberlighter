def highpass_filter(signal, cutoff=0.01, fs=5, order=4):
    """Remove slow drift below cutoff frequency"""
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)

def double_exponential(t, A1, tau1, A2, tau2, baseline):
    """
    Double exponential decay
    
    Parameters:
    -----------
    t : time
    A1, A2 : amplitudes of two components
    tau1, tau2 : time constants (tau1 < tau2 usually)
    baseline : offset
    """
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + baseline


def fit_double_exp_bleaching(signal, time):
    """
    Fit double exponential to bleaching
    """
    
    # Initial guess
    p0 = [
        signal[0] * 0.3,  # A1 (fast component amplitude)
        30,               # tau1 (fast decay ~30s)
        signal[0] * 0.7,  # A2 (slow component amplitude)
        300,              # tau2 (slow decay ~5min)
        signal[-1]        # baseline
    ]
    
    # Bounds (keep parameters reasonable)
    bounds = (
        [0, 1, 0, 60, 0],           # Lower bounds
        [np.inf, 120, np.inf, 3600, np.inf]  # Upper bounds
    )
    
    try:
        # Fit
        params, covariance = curve_fit(
            double_exponential, 
            time, 
            signal,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        
        A1, tau1, A2, tau2, baseline = params
        
        # Generate fitted curve
        fitted = double_exponential(time, *params)
        
        print(f"\nDouble Exponential Fit Results:")
        print(f"  Fast component:")
        print(f"    Amplitude (A1): {A1:.2f}")
        print(f"    Time constant (τ1): {tau1:.2f} seconds")
        print(f"  Slow component:")
        print(f"    Amplitude (A2): {A2:.2f}")
        print(f"    Time constant (τ2): {tau2:.2f} seconds ({tau2/60:.1f} min)")
        print(f"  Baseline: {baseline:.2f}")
        
        return {
            'params': params,
            'fitted': fitted,
            'A1': A1, 'tau1': tau1,
            'A2': A2, 'tau2': tau2,
            'baseline': baseline,
        }
        
    except Exception as e:
        print(f"Fit failed: {e}")
        return None
ignal_detrend = detrend(signal)
iso_detrend = detrend(iso_interp)

plt.figure(figsize=(15, 10))
plt.plot(time_signal, signal, alpha=0.5, label='Raw data')
plt.plot(time_signal, iso_interp, alpha=0.5, label='Raw data')
plt.plot(time_signal, signal_detrend, label='Detrended GCaMP')
plt.plot(time_signal, iso_detrend, label='Detrended 405')

plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')
plt.legend()
plt.title('Detrending')
plt.grid(True, alpha=0.3)
plt.show()

import numpy as np

def hamming_window(n):
    return 0.54 - 0.46 * np.cos(2*np.pi*np.arange(n) / (n - 1))

def fir_lowpass(numtaps, cutoff_norm):
    h = np.zeros(numtaps)
    M = numtaps - 1

    for n in range(numtaps):
        if n == M // 2:
            h[n] = 2 * cutoff_norm
        else:
            h[n] = (np.sin(2 * np.pi * cutoff_norm *(n - M/2)) / (np.pi * (n - M/2)))

    return h

def fir_bandpass(numtaps, f1_norm, f2_norm):

    h_high = fir_lowpass(numtaps, f2_norm)
    h_low = fir_lowpass(numtaps, f1_norm)

    h_low_hp = -h_low
    h_low_hp[numtaps//2] += 1

    return h_high + h_low_hp

def fir_window(numtaps, f1_norm, f2_norm, window_type = "hamming"):

    h = fir_bandpass(numtaps, f1_norm, f2_norm)

    if window_type == "hamming":
        window = hamming_window(numtaps)
    else:
        raise ValueError(f"Not known window type")

    return (h * window)

def design_third_fir(fs, f1, f2, numtaps = 1011, window_type="hamming"):

    f1_norm = f1 / fs
    f2_norm = f2 / fs

    f1_norm = max(f1_norm, 0.001)
    f2_norm = min(f2_norm, 0.495)

    h = fir_window(numtaps, f1_norm, f2_norm, window_type)

    return h.astype(np.float32)


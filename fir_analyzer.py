import numpy as np
from scipy.signal import lfilter
from fir_filters import design_third_fir

def analyze_frame_fir(frame, fir_bank):
    band_powers = []

    for h in fir_bank:
        y = lfilter(h, 1.0, frame)
        p = float(np.mean(y * y))
        band_powers.append(p)

    return band_powers

def setup_fir_analysis(fs, bands, numtaps = 1011, window_type = "hamming"):
    fir_bank = []

    for (_, f1, f2) in bands:
        h = design_third_fir(fs, f1, f2, numtaps, window_type)
        fir_bank.append(h)

    return fir_bank

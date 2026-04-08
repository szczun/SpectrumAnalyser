import numpy as np

def analyze_frame_fft(frame, w, idx_map):
    X = np.fft.rfft(frame * w)
    P = (np.abs(X)**2) / len(frame)
    
    band_powers = []
    for (i1, i2) in idx_map:
        p = P[i1:(i2 + 1)].mean() if i2 >= i1 else 0.0
        band_powers.append(p)
    
    return band_powers

def setup_fft_analysis(N, fs, bands):
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    idx_map = []
    for (_, f1, f2) in bands:
        i1 = int(np.floor(np.interp(f1, freqs, np.arange(len(freqs)))))
        i2 = int(np.ceil(np.interp(f2, freqs, np.arange(len(freqs)))))
        i1 = max(i1, 0)
        i2 = min(i2, len(freqs) - 1)
        if i2 <= i1:
            i2 = min(i1 + 1, len(freqs) - 1)
        idx_map.append((i1, i2))
    
    return freqs, idx_map

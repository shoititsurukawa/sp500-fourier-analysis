import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("SP500.csv")
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.dropna(subset=['SP500'])

y = df['SP500'].values.astype(float)

# Remove mean (remove DC)
y = y - np.mean(y)

# --------------------------------
# Welch Power Spectral Density
# fs = 1 sample/day
# --------------------------------
fs = 1

f, Pxx = welch(
    y,
    fs=fs,
    window='hann',
    nperseg=min(512, len(y)//4),
    scaling='spectrum'
)

# --------------------------------
# Find dominant peaks
# --------------------------------
peaks, _ = find_peaks(Pxx)

# strongest 10 peaks
N = 10
idx = peaks[np.argsort(Pxx[peaks])[-N:]]
idx = idx[np.argsort(f[idx])]

print("\nDominant Spectral Components")
print("------------------------------------------")
print("n   Frequency(cycles/day)   Period(days)   Power")
print("------------------------------------------")

for i,k in enumerate(idx,1):

    freq = f[k]
    T = 1/freq

    print(
        f"{i:2d}   {freq:12.6f}       "
        f"{T:8.2f}      {Pxx[k]:10.4e}"
    )

# --------------------------------
# Plot time-domain signal
# --------------------------------
plt.figure(figsize=(12,5))
plt.plot(df['observation_date'], y)
plt.title("Mean-Removed S&P500")
plt.grid(True)
plt.show()


# --------------------------------
# Plot frequency-domain PSD
# --------------------------------
plt.figure(figsize=(12,6))
plt.semilogy(f, Pxx)

# Mark dominant peaks
plt.plot(f[idx], Pxx[idx], 'o')

plt.xlabel("Frequency (cycles/day)")
plt.ylabel("PSD")
plt.title("Welch Power Spectral Density")
plt.grid(True)
plt.show()
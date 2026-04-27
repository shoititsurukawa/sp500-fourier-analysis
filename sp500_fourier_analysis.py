import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load data
# --------------------------
df = pd.read_csv("SP500.csv")
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.dropna(subset=['SP500'])

y = df['SP500'].values.astype(float)
t = np.arange(len(y))

# Remove mean (DC term)
a0 = np.mean(y)
yc = y - a0

# --------------------------
# FFT
# --------------------------
Y = np.fft.fft(yc)
freq = np.fft.fftfreq(len(y))

# Positive frequencies only
half = len(y)//2
mag = np.abs(Y[:half])

# Ignore DC
mag[0] = 0

# Number of harmonics to keep
N = 10

# indices of N strongest harmonics
harmonics = np.argsort(mag)[-N:]
harmonics = harmonics[np.argsort(harmonics)]  # sort by frequency

# Reconstruction
y_fourier = np.ones_like(y)*a0

print("Harmonics:")
print("-------------------------------")
print("n   Frequency      Amplitude")
print("-------------------------------")

for i,k in enumerate(harmonics,1):

    A = 2*np.abs(Y[k])/len(y)
    phi = np.angle(Y[k])
    f = freq[k]

    print(f"{i:2d}  {f:10.6f}   {A:10.4f}")

    y_fourier += A*np.cos(2*np.pi*f*t + phi)

# --------------------------
# Plot
# --------------------------
plt.figure(figsize=(12,6))
plt.plot(df['observation_date'], y,
         label='Original', alpha=0.7)

plt.plot(df['observation_date'], y_fourier,
         linewidth=3,
         label=f'{N}-Term Fourier Approximation')

plt.title('Fourier Series Approximation of S&P500')
plt.xlabel('Date')
plt.ylabel('Index')
plt.legend()
plt.grid(True)
plt.show()
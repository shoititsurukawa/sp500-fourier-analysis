import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("SP500.csv")
df['observation_date'] = pd.to_datetime(df['observation_date'])
df = df.dropna(subset=['SP500'])

y = df['SP500'].values.astype(float)

# Remove mean / optional detrend
y = y - np.mean(y)

# -------------------------
# Continuous Wavelet Transform
# -------------------------

# Scales (controls periods examined)
scales = np.arange(1,512)

# Morlet wavelet
coeffs, freqs = pywt.cwt(
    y,
    scales,
    'morl',
    sampling_period=1  # 1 sample/day
)

# Wavelet power
power = np.abs(coeffs)**2

# Convert pseudo-frequency -> period
period = 1/freqs


# -------------------------
# Scalogram
# -------------------------
plt.figure(figsize=(10,5))

plt.contourf(
    df['observation_date'],
    period,
    power,
    100
)

plt.yscale('log')
plt.gca().invert_yaxis()

plt.colorbar(label='Wavelet Power')

plt.xlabel('Date')
plt.ylabel('Period (days)')
plt.title('Morlet Wavelet Scalogram — S&P500')

plt.tight_layout()
plt.show()
from rtlsdr import RtlSdr
import numpy as np

# Configure RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6     # 2.4 MS/s
sdr.center_freq = 88.0e6    # 88.0 MHz
sdr.gain = 'auto'           # Automatic gain

# Capture samples
num_samples = 2_400_000     # ~1 second at 2.4MS/s
print("Capturing IQ data...")
samples = sdr.read_samples(num_samples)

# Save to .npy file
np.save("iq_88MHz.npy", samples)
print("Saved to iq_88MHz.npy")

# Cleanup
sdr.close()

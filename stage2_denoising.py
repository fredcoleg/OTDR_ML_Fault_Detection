import numpy as np
import matplotlib

matplotlib.use('Agg')  # Headless mode for seamless plot generation
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter


def wavelet_denoise(data, wavelet='db4', level=3):
    # Discrete Wavelet Transform (DWT)
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Estimate noise standard deviation from detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    # Soft thresholding
    coeffs_thresh = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]

    # Reconstruct 1D signal
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    return reconstructed[:len(data)]


def run_stage2_pipeline():
    try:
        data = np.loadtxt("synthetic_otdr_trace.csv", delimiter=",", skiprows=1)
    except OSError:
        print("Error: 'synthetic_otdr_trace.csv' not found. Run stage1_synthetic_trace.py first!")
        return

    dist = data[:, 0]
    clean_trace = data[:, 1]
    noisy_trace = data[:, 2]

    # Apply Denoising Algorithms
    wavelet_trace = wavelet_denoise(noisy_trace, wavelet='db4', level=3)
    savgol_trace = savgol_filter(noisy_trace, window_length=51, polyorder=3)

    # Compute Root Mean Square Error (RMSE)
    rmse_noisy = np.sqrt(np.mean((noisy_trace - clean_trace) ** 2))
    rmse_wavelet = np.sqrt(np.mean((wavelet_trace - clean_trace) ** 2))
    rmse_savgol = np.sqrt(np.mean((savgol_trace - clean_trace) ** 2))

    print("--- Denoising Performance Evaluation ---")
    print(f"Raw Noisy Input RMSE:      {rmse_noisy:.4f} dB")
    print(f"Wavelet (db4) Filter RMSE: {rmse_wavelet:.4f} dB")
    print(f"Savitzky-Golay Filter RMSE:{rmse_savgol:.4f} dB")

    # Save Comparison Plot
    plt.figure(figsize=(11, 5))
    plt.plot(dist, noisy_trace, color='lightgray', label='Raw Noisy Input')
    plt.plot(dist, clean_trace, color='black', linestyle='--', linewidth=1.5, label='Ground Truth Target')
    plt.plot(dist, wavelet_trace, color='blue', linewidth=1.2, label='Wavelet (db4) Denoised')
    plt.plot(dist, savgol_trace, color='red', linewidth=1.2, label='Savitzky-Golay Denoised')

    plt.title('Stage 2: Software Denoising vs. Feature Preservation')
    plt.xlabel('Distance (km)')
    plt.ylabel('Optical Power (dB)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("stage2_plot.png", dpi=300)
    print("Plot successfully saved as 'stage2_plot.png'")


if __name__ == "__main__":
    run_stage2_pipeline()
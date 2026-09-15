import numpy as np
import matplotlib

matplotlib.use('Agg')  # Headless mode: zero GUI dependencies required
import matplotlib.pyplot as plt


def generate_otdr_dataset(
        fiber_length_km=20.0,
        resolution_m=1.0,
        attenuation_db_km=0.2,  # Standard SMF-28 loss at 1550 nm
        snr_db=12.0  # Low SNR simulates un-averaged single-shot hardware traces
):
    num_points = int((fiber_length_km * 1000) / resolution_m)
    distance_km = np.linspace(0, fiber_length_km, num_points)

    clean_power_db = 0.0 - (attenuation_db_km * distance_km)

    # Event 1: Connector Reflection Spike + Loss at 5.0 km
    idx_5km = int(5000 / resolution_m)
    spike_len = int(15 / resolution_m)
    clean_power_db[idx_5km: idx_5km + spike_len] += 10.0 * np.exp(-np.linspace(0, 3, spike_len))
    clean_power_db[idx_5km:] -= 0.4

    # Event 2: Non-Reflective Fusion Splice Drop at 12.0 km
    idx_12km = int(12000 / resolution_m)
    clean_power_db[idx_12km:] -= 0.25

    # Event 3: Fiber Cut / End Reflection Spike at 17.5 km
    idx_cut = int(17500 / resolution_m)
    end_spike_len = int(20 / resolution_m)
    clean_power_db[idx_cut: idx_cut + end_spike_len] += 15.0 * np.exp(-np.linspace(0, 3, end_spike_len))
    clean_power_db[idx_cut:] = -45.0

    # Inject Additive White Gaussian Noise (AWGN)
    power_linear = 10 ** (clean_power_db / 10.0)
    signal_power = np.mean(power_linear[:idx_cut])
    noise_variance = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_variance), size=num_points)

    noisy_power_linear = np.maximum(power_linear + noise, 1e-5)
    noisy_power_db = 10 * np.log10(noisy_power_linear)

    return distance_km, clean_power_db, noisy_power_db


if __name__ == "__main__":
    dist, clean_trace, noisy_trace = generate_otdr_dataset(snr_db=12.0)

    dataset = np.column_stack((dist, clean_trace, noisy_trace))
    np.savetxt("synthetic_otdr_trace.csv", dataset, delimiter=",", header="Distance_km,Clean_dB,Noisy_dB", comments="")
    print("Dataset saved successfully as 'synthetic_otdr_trace.csv'")

    plt.figure(figsize=(10, 5))
    plt.plot(dist, noisy_trace, color='gray', alpha=0.6, label='Raw Noisy Input (Low Averaging)')
    plt.plot(dist, clean_trace, color='blue', linewidth=1.5, label='Ground Truth Target')
    plt.title('Stage 1: Synthetic OTDR Trace with AWGN Injection')
    plt.xlabel('Distance (km)')
    plt.ylabel('Optical Power (dB)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig("stage1_plot.png", dpi=300)
    print("Plot successfully saved as 'stage1_plot.png'")
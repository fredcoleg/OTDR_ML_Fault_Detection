import numpy as np
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class OTDRFaultClassifier1D(nn.Module):
    def __init__(self, num_classes=4):
        super(OTDRFaultClassifier1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 25, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def suppress_adjacent_events(events, distance_radius=0.5):
    """Non-Maximum Suppression (NMS): Keeps only the highest confidence peak per region."""
    if not events:
        return []

    # Sort events by confidence descending
    sorted_events = sorted(events, key=lambda x: x[2], reverse=True)
    filtered_events = []

    for current in sorted_events:
        c_dist, c_type, c_conf = current
        # If no already-selected event is within radius, keep it
        if not any(abs(c_dist - f[0]) < distance_radius for f in filtered_events):
            filtered_events.append(current)

    return sorted(filtered_events, key=lambda x: x[0])


def run_stage4_inference():
    try:
        data = np.loadtxt("synthetic_otdr_trace.csv", delimiter=",", skiprows=1)
    except OSError:
        print("Error: 'synthetic_otdr_trace.csv' not found.")
        return

    dist = data[:, 0]
    noisy_trace = data[:, 2]
    filtered_trace = savgol_filter(noisy_trace, window_length=51, polyorder=3)
    trace_grad = np.gradient(filtered_trace)

    model = OTDRFaultClassifier1D()
    model.load_state_dict(torch.load("otdr_1dcnn_model.pth"))
    model.eval()

    window_size, step_size = 100, 10
    class_names = {0: "Normal", 1: "Reflective Connector Spike", 2: "Fusion Splice Drop", 3: "Fiber Cut / Termination"}

    raw_detections = []
    confidence_threshold = 0.75  # Filters out noise guesses (< 74%) while keeping true events (>= 75%)

    with torch.no_grad():
        for i in range(0, len(filtered_trace) - window_size, step_size):
            window_sig = filtered_trace[i: i + window_size]
            window_grad = trace_grad[i: i + window_size]
            center_dist = dist[i + (window_size // 2)]

            norm_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 0.1)
            feature_window = np.stack([norm_sig, window_grad], axis=0)
            tensor_in = torch.tensor(feature_window, dtype=torch.float32).unsqueeze(0)

            probs = torch.softmax(model(tensor_in), dim=1)
            conf, pred = torch.max(probs, dim=1)
            pred, conf = pred.item(), conf.item()

            if pred != 0 and conf >= confidence_threshold:
                raw_detections.append((center_dist, class_names[pred], conf))

    # Apply NMS peak clustering to suppress nearby duplicate sliding-window triggers
    clean_events = suppress_adjacent_events(raw_detections, distance_radius=0.5)

    print("--- Stage 4: Automated Event Localization Report ---")
    if not clean_events:
        print("No optical faults detected above confidence threshold.")
    else:
        for distance_km, event_type, conf in clean_events:
            print(f"Distance: {distance_km:6.2f} km  |  Detected: {event_type:<28} | Confidence: {conf * 100:.1f}%")

    # Generate diagnostic plot
    plt.figure(figsize=(11, 5))
    plt.plot(dist, noisy_trace, color='gray', alpha=0.5, label='Raw Signal')
    plt.plot(dist, filtered_trace, color='blue', linewidth=1.0, label='Filtered Signal')

    for distance_km, event_type, conf in clean_events:
        plt.axvline(x=distance_km, color='red', linestyle='--', alpha=0.8)
        plt.text(distance_km + 0.1, -15, f"{event_type}\n@ {distance_km:.2f} km ({conf * 100:.0f}%)",
                 color='red', fontweight='bold', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    plt.title('Stage 4: Automated Optical Fault Localization Map')
    plt.xlabel('Distance (km)')
    plt.ylabel('Optical Power (dB)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("stage4_plot.png", dpi=300)
    print("\nClean diagnostic plot saved as 'stage4_plot.png'")


if __name__ == "__main__":
    run_stage4_inference()
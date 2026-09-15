import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==========================================
# STAGE 3: 1D-CNN MODEL DEFINITION
# ==========================================
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


# ==========================================
# STAGE 1: SYNTHETIC TRACE GENERATION
# ==========================================
def run_stage1_generate_trace(filename="synthetic_otdr_trace.csv"):
    print("[Stage 1/4] Generating synthetic OTDR fiber trace...")
    distance = np.linspace(0, 20, 2000)  # 20 km fiber link (10m resolution)
    alpha = 0.2  # Fiber attenuation coefficient (0.2 dB/km)

    # Baseline linear attenuation
    power = -alpha * distance

    # Event 1: Reflective Connector Spike at ~5.0 km
    spike_idx = 500
    power[spike_idx:spike_idx + 5] += np.array([3.0, 8.0, 5.0, 2.0, 0.5])

    # Event 2: Non-Reflective Fusion Splice Drop at ~12.0 km
    splice_idx = 1200
    power[splice_idx:] -= 1.5  # 1.5 dB splice loss

    # Event 3: Fiber Cut / End of Fiber at ~17.5 km
    cut_idx = 1750
    power[cut_idx:cut_idx + 8] += np.array([1.5, 4.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    power[cut_idx + 8:] = -45.0  # Noise floor post-cut

    # Add Rayleigh backscatter Gaussian noise
    noise = np.random.normal(0, 0.35, size=len(distance))
    noisy_power = power + noise

    data = np.column_stack((distance, power, noisy_power))
    header = "Distance_km,Ideal_Power_dB,Noisy_Power_dB"
    np.savetxt(filename, data, delimiter=",", header=header, comments="")
    print(f" -> Trace successfully written to '{filename}'.")


# ==========================================
# STAGE 2 & 3: TRACE SLICING & MODEL TRAINING
# ==========================================
def create_dataset_from_trace(trace_file="synthetic_otdr_trace.csv", window_size=100, step_size=5):
    """Extracts 2-channel features directly from sliding trace windows for exact feature alignment."""
    data = np.loadtxt(trace_file, delimiter=",", skiprows=1)
    dist = data[:, 0]
    noisy_trace = data[:, 2]
    filtered_trace = savgol_filter(noisy_trace, window_length=51, polyorder=3)
    trace_grad = np.gradient(filtered_trace)

    X, y = [], []
    for i in range(0, len(filtered_trace) - window_size, step_size):
        window_sig = filtered_trace[i: i + window_size]
        window_grad = trace_grad[i: i + window_size]
        center_dist = dist[i + (window_size // 2)]

        norm_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 0.1)
        feature_window = np.stack([norm_sig, window_grad], axis=0)

        # Assign ground truth labels based on trace distance
        if 4.8 <= center_dist <= 5.2:
            label = 1  # Reflective Connector Spike
        elif 11.8 <= center_dist <= 12.2:
            label = 2  # Fusion Splice Drop
        elif 17.3 <= center_dist <= 17.7:
            label = 3  # Fiber Cut / Termination
        else:
            label = 0  # Normal Fiber

        X.append(feature_window)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Class balancing via oversampling
    X_balanced, y_balanced = [], []
    normal_idx = np.where(y == 0)[0]
    X_balanced.extend(X[normal_idx])
    y_balanced.extend(y[normal_idx])

    for c in [1, 2, 3]:
        c_idx = np.where(y == c)[0]
        if len(c_idx) > 0:
            repeat_factor = len(normal_idx) // len(c_idx)
            X_balanced.extend(np.tile(X[c_idx], (repeat_factor, 1, 1)))
            y_balanced.extend(np.tile(y[c_idx], repeat_factor))

    return torch.tensor(np.array(X_balanced), dtype=torch.float32), torch.tensor(np.array(y_balanced), dtype=torch.long)


def run_stage2_3_train_model(trace_file="synthetic_otdr_trace.csv", model_path="otdr_1dcnn_model.pth", epochs=25):
    print("\n[Stage 2 & 3/4] Preprocessing features & training 1D-CNN classifier...")
    X, y = create_dataset_from_trace(trace_file)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = OTDRFaultClassifier1D()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        if epoch % 5 == 0 or epoch == epochs:
            epoch_loss = running_loss / len(dataset)
            print(f" -> Epoch {epoch:02d}/{epochs:02d} | Training Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f" -> Model weights successfully saved to '{model_path}'.")


# ==========================================
# STAGE 4: AUTOMATED INFERENCE & NMS REPORT
# ==========================================
def suppress_adjacent_events(events, distance_radius=0.6):
    """Non-Maximum Suppression (NMS) to merge close detections into a single peak."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda x: x[2], reverse=True)
    filtered_events = []

    for current in sorted_events:
        c_dist, c_type, c_conf = current
        if not any(abs(c_dist - f[0]) < distance_radius for f in filtered_events):
            filtered_events.append(current)

    return sorted(filtered_events, key=lambda x: x[0])


def run_stage4_inference(trace_file="synthetic_otdr_trace.csv", model_path="otdr_1dcnn_model.pth"):
    print("\n[Stage 4/4] Running automated event localization & diagnostic visualization...")

    data = np.loadtxt(trace_file, delimiter=",", skiprows=1)
    dist = data[:, 0]
    noisy_trace = data[:, 2]
    filtered_trace = savgol_filter(noisy_trace, window_length=51, polyorder=3)
    trace_grad = np.gradient(filtered_trace)

    model = OTDRFaultClassifier1D()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    window_size, step_size = 100, 10
    class_names = {0: "Normal", 1: "Reflective Connector Spike", 2: "Fusion Splice Drop", 3: "Fiber Cut / Termination"}
    raw_detections = []
    confidence_threshold = 0.85  # Precision threshold for trained trace windows

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

    clean_events = suppress_adjacent_events(raw_detections, distance_radius=0.6)

    print("\n========================================================")
    print("      STAGE 4: AUTOMATED EVENT LOCALIZATION REPORT     ")
    print("========================================================")
    if not clean_events:
        print("No optical faults detected above confidence threshold.")
    else:
        for distance_km, event_type, conf in clean_events:
            print(f"Distance: {distance_km:6.2f} km  |  Detected: {event_type:<28} | Confidence: {conf * 100:.1f}%")
    print("========================================================\n")

    # Diagnostic Plot Generation
    plt.figure(figsize=(11, 5))
    plt.plot(dist, noisy_trace, color='gray', alpha=0.5, label='Raw Signal')
    plt.plot(dist, filtered_trace, color='blue', linewidth=1.0, label='Filtered Signal')

    for distance_km, event_type, conf in clean_events:
        plt.axvline(x=distance_km, color='red', linestyle='--', alpha=0.8)
        plt.text(distance_km + 0.1, -15, f"{event_type}\n@ {distance_km:.2f} km ({conf * 100:.0f}%)",
                 color='red', fontweight='bold', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    plt.title('Stage 4: End-to-End Automated Optical Fault Localization Map')
    plt.xlabel('Distance (km)')
    plt.ylabel('Optical Power (dB)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("stage4_plot.png", dpi=300)
    print("Clean diagnostic plot saved as 'stage4_plot.png'")


# ==========================================
# PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    run_stage1_generate_trace()
    run_stage2_3_train_model()
    run_stage4_inference()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split


def create_segmented_dataset(window_size=100, step_size=10):
    try:
        data = np.loadtxt("synthetic_otdr_trace.csv", delimiter=",", skiprows=1)
    except OSError:
        print("Error: 'synthetic_otdr_trace.csv' not found. Run Stage 1 first!")
        return None, None

    dist = data[:, 0]
    noisy_trace = data[:, 2]
    filtered_trace = savgol_filter(noisy_trace, window_length=51, polyorder=3)

    # Derivative feature (dP/dx) isolates sharp spikes and drops from linear attenuation
    trace_grad = np.gradient(filtered_trace)

    X, y = [], []
    for i in range(0, len(filtered_trace) - window_size, step_size):
        window_sig = filtered_trace[i: i + window_size]
        window_grad = trace_grad[i: i + window_size]
        center_dist = dist[i + (window_size // 2)]

        # Channel 0: Signal relative to mean | Channel 1: Derivative
        norm_sig = (window_sig - np.mean(window_sig)) / (np.std(window_sig) + 0.1)
        feature_window = np.stack([norm_sig, window_grad], axis=0)

        # Label strictly by trace distance in kilometers
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

    # Oversample minority classes to create a balanced dataset
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

    return np.array(X_balanced, dtype=np.float32), np.array(y_balanced, dtype=np.int64)


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


def train_stage3_model():
    X, y = create_segmented_dataset()
    if X is None: return

    counts = np.bincount(y, minlength=4)
    print(f"Balanced Dataset: Normal={counts[0]}, Connector={counts[1]}, Splice={counts[2]}, Cut={counts[3]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = OTDRFaultClassifier1D()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training 2-Channel 1D-CNN Model...")
    model.train()
    for epoch in range(1, 26):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train))
        loss = criterion(outputs, torch.tensor(y_train))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/25] - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "otdr_1dcnn_model.pth")
    print("Model saved successfully as 'otdr_1dcnn_model.pth'")


if __name__ == "__main__":
    train_stage3_model()
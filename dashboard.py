import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="OTDR ML Fault Detection Dashboard",
    page_icon="⚡",
    layout="wide"
)


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
# HELPER FUNCTIONS & INFERENCE LOGIC
# ==========================================
@st.cache_resource
def load_trained_model(model_path="otdr_1dcnn_model.pth"):
    """Loads pre-trained 1D-CNN model weights."""
    model = OTDRFaultClassifier1D()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, True
    return None, False


def generate_default_trace():
    """Generates a default 20 km synthetic trace if no file is uploaded."""
    distance = np.linspace(0, 20, 2000)
    alpha = 0.2
    power = -alpha * distance

    # Event 1: Reflective Spike
    power[500:505] += np.array([3.0, 8.0, 5.0, 2.0, 0.5])
    # Event 2: Splice Loss
    power[1200:] -= 1.5
    # Event 3: Fiber Cut
    power[1750:1758] += np.array([1.5, 4.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    power[1758:] = -45.0

    noisy_power = power + np.random.normal(0, 0.35, size=len(distance))
    return pd.DataFrame({"Distance_km": distance, "Ideal_Power_dB": power, "Noisy_Power_dB": noisy_power})


def suppress_adjacent_events(events, distance_radius=0.6):
    """Non-Maximum Suppression (NMS) peak clustering."""
    if not events:
        return []
    sorted_events = sorted(events, key=lambda x: x[2], reverse=True)
    filtered_events = []

    for current in sorted_events:
        c_dist, c_type, c_conf = current
        if not any(abs(c_dist - f[0]) < distance_radius for f in filtered_events):
            filtered_events.append(current)

    return sorted(filtered_events, key=lambda x: x[0])


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🔧 Pipeline Controls")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload OTDR Trace CSV", type=["csv"])

st.sidebar.subheader("Inference Hyperparameters")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.50, max_value=0.99, value=0.85, step=0.01)
nms_radius = st.sidebar.slider("NMS Clustering Radius (km)", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
savgol_window = st.sidebar.slider("Savitzky-Golay Filter Window", min_value=11, max_value=101, value=51, step=2)

st.sidebar.markdown("---")
st.sidebar.info("Model Status: `otdr_1dcnn_model.pth`")

# ==========================================
# MAIN DASHBOARD LAYOUT
# ==========================================
st.title("⚡ OTDR Machine Learning Optical Fault Detection System")
st.markdown(
    "Automated fiber optic fault detection, classification, and event localization using a 1D Convolutional Neural Network.")

# Load Data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom OTDR trace file loaded successfully.")
else:
    df = generate_default_trace()
    st.info("Using baseline 20 km synthetic OTDR trace. (Upload a CSV file in the sidebar to analyze custom traces).")

# Process Signal
dist = df["Distance_km"].values
noisy_trace = df["Noisy_Power_dB"].values
filtered_trace = savgol_filter(noisy_trace, window_length=savgol_window, polyorder=3)
trace_grad = np.gradient(filtered_trace)

# Load Model
model, model_loaded = load_trained_model()

if not model_loaded:
    st.error(
        "Error: Model weights `otdr_1dcnn_model.pth` not found. Please run `python main.py` once to train and save the model.")
    st.stop()

# Perform Inference
window_size, step_size = 100, 10
class_names = {0: "Normal", 1: "Reflective Connector Spike", 2: "Fusion Splice Drop", 3: "Fiber Cut / Termination"}
raw_detections = []

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

clean_events = suppress_adjacent_events(raw_detections, distance_radius=nms_radius)

# ==========================================
# METRICS DASHBOARD
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Fiber Distance", f"{dist[-1]:.2f} km")
col2.metric("Detected Fault Events", f"{len(clean_events)}")
col3.metric("Signal Noise Floor", f"{np.min(noisy_trace):.1f} dB")
col4.metric("Classification Model", "1D-CNN PyTorch")

st.markdown("---")

# ==========================================
# DIAGNOSTIC PLOT & REPORT TABLE
# ==========================================
col_left, col_right = st.columns([7, 5])

with col_left:
    st.subheader("📈 Diagnostic Optical Power Trace")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dist, noisy_trace, color='gray', alpha=0.4, label='Raw Noisy Signal')
    ax.plot(dist, filtered_trace, color='#0055ff', linewidth=1.2, label='Filtered Signal')

    for distance_km, event_type, conf in clean_events:
        ax.axvline(x=distance_km, color='red', linestyle='--', alpha=0.8)
        ax.text(distance_km + 0.1, -15, f"{event_type}\n@ {distance_km:.2f} km ({conf * 100:.0f}%)",
                color='red', fontweight='bold', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    ax.set_title('Automated Optical Fault Localization Map')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Optical Power (dB)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

with col_right:
    st.subheader("📋 Event Localization Report")
    if clean_events:
        report_data = []
        for distance_km, event_type, conf in clean_events:
            report_data.append({
                "Distance (km)": f"{distance_km:.2f}",
                "Event Type": event_type,
                "Confidence": f"{conf * 100:.1f}%"
            })
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)

        # CSV Export Option
        csv_data = report_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Event Report CSV",
            data=csv_data,
            file_name="otdr_fault_report.csv",
            mime="text/csv"
        )
    else:
        st.warning("No optical fault events detected above the current confidence threshold.")
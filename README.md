# ⚡ OTDR Machine Learning Optical Fault Detection System

An end-to-end deep learning framework and interactive Streamlit web application for automated fiber optic trace analysis, fault classification, and event localization using a 1D Convolutional Neural Network (1D-CNN).

---

## 📌 Project Overview

Optical Time-Domain Reflectometers (OTDR) capture power loss and backscatter traces to diagnose physical anomalies in optical fiber distribution networks. Manual evaluation of raw traces can be time-consuming and prone to subjective error—especially under low Signal-to-Noise Ratios (SNR).

This project automates trace diagnostics through a modular 4-stage pipeline:

1. **Data Generation & Ingestion:** Simulates realistic 20 km single-mode fiber OTDR traces complete with Rayleigh backscatter, attenuation slopes, splice loss events, reflective connector spikes, and structural fiber breaks.
2. **Signal Denoising:** Applies a Savitzky-Golay smoothing filter to attenuate high-frequency noise while preserving event edge transitions.
3. **Deep Learning Feature Extraction:** Employs a 1D-CNN trained in PyTorch to classify localized trace windows into event categories.
4. **Event Localization:** Applies Non-Maximum Suppression (NMS) to resolve overlapping candidate predictions and map exact fault distances (in kilometers).

---

## 🏗️ 1D-CNN Model Architecture

The deep learning model operates on sliding 1D trace sequence windows:

| Layer | Layer Type | Specs & Parameters |
| :--- | :--- | :--- |
| **Input** | 1D Window | Sequence length = 128 points |
| **Conv Block 1** | Conv1D + BatchNorm + ReLU + MaxPool | 16 Filters, Kernel = 5, Stride = 1, MaxPool = 2 |
| **Conv Block 2** | Conv1D + BatchNorm + ReLU + MaxPool | 32 Filters, Kernel = 3, Stride = 1, MaxPool = 2 |
| **Conv Block 3** | Conv1D + BatchNorm + ReLU + MaxPool | 64 Filters, Kernel = 3, Stride = 1, MaxPool = 2 |
| **Classifier** | Linear + Dropout (0.3) + Linear | Fully connected dense layers |
| **Output** | Softmax | Multi-class probability distribution |

---

## 📁 Repository Structure

```text
OTDR_ML_Fault_Detection/
├── .gitignore                  # Git ignore directives
├── dashboard.py                # Streamlit interactive web GUI
├── main.py                     # Pipeline execution script
├── otdr_1dcnn_model.pth        # Trained PyTorch model checkpoint
├── stage1_synthetic_trace.py   # OTDR trace generator engine
├── stage2_denoising.py         # Savitzky-Golay signal filtering
├── stage3_model_training.py    # PyTorch 1D-CNN training pipeline
├── stage4_fault_detection.py   # NMS clustering & localization engine
└── synthetic_otdr_trace.csv    # Sample 20 km baseline OTDR trace dataset
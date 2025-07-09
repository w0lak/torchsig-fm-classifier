import numpy as np
import torch
from torchsig.models.iq_models.densenet.densenet1d import densenet121_1d
from torchsig.transforms.functional import iq_normalize
import os

# Load captured IQ samples
iq_data = np.load("iq_88MHz.npy")  # shape: (N, 2)
print(f"Loaded IQ shape: {iq_data.shape}")

# Convert to tensor and normalize
iq_tensor = torch.from_numpy(iq_data.T).unsqueeze(0).float()  # shape: (1, 2, N)
iq_tensor = iq_normalize(iq_tensor)

# Load model
model = densenet121_1d(num_classes=26)  # 26 = number of classes in Sig53
model.eval()

# Load weights if available (optional):
# model.load_state_dict(torch.load("model_weights.pth"))

# Inference
with torch.no_grad():
    logits = model(iq_tensor)
    probs = torch.softmax(logits, dim=1)
    predicted_index = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_index].item()

# Map index to class name (Sig53 example)
sig53_classes = [
    "WFM", "AM-DSB", "AM-SSB", "AM", "QPSK", "BPSK", "8PSK", "16QAM", "64QAM",
    "GFSK", "CPFSK", "PAM4", "QAM", "GMSK", "OQPSK", "FM", "FSK", "ASK", "Burst", 
    "DSSS", "TDMA", "OFDM", "Radar", "LTE", "GSM", "Unknown"
]

predicted_class = sig53_classes[predicted_index] if predicted_index < len(sig53_classes) else "Unknown"

print(f"Predicted Class: {predicted_class} (confidence: {confidence:.2f})")


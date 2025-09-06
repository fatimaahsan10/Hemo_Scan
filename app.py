import os
import io
import math
import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

import torchvision
from torchvision import transforms
import gradio as gr

# -----------------------------
# Utils
# -----------------------------

IM_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_gdl(hb_gl: float) -> float:
    return round(float(hb_gl) / 10.0, 1)

def risk_from_hbdl(hbdl: float) -> str:
    if hbdl < 11.0:
        return "High (Anemia likely)"
    elif hbdl <= 14.0:
        return "Medium (Borderline)"
    else:
        return "Low (Normal)"

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(img.convert("RGB")).unsqueeze(0)

# -----------------------------
# Model (must match training)
# -----------------------------

class MultiHeadMobileNet(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        # Load backbone
        try:
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            base = torchvision.models.mobilenet_v3_small(weights=weights)
        except Exception:
            base = torchvision.models.mobilenet_v3_small(pretrained=True)

        self.backbone = base.features  # conv features
        self.pool = nn.AdaptiveAvgPool2d(1)
        # in_features is the input dim of the first Linear layer in the original classifier
        in_features = base.classifier[0].in_features  # typically 576
        hidden = 1024
        self.feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.Hardswish(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(hidden, num_classes)
        self.reg_head = nn.Linear(hidden, 1)

        # For Grad-CAM: cache activations & grads from the last conv feature map
        self._acts = None
        self._grads = None
        # Find last conv layer inside backbone for CAM
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is not None:
            def fwd_hook(module, inp, out):
                self._acts = out.detach()
            def bwd_hook(module, grad_in, grad_out):
                self._grads = grad_out[0].detach()
            last_conv.register_forward_hook(fwd_hook)
            last_conv.register_full_backward_hook(bwd_hook)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        pooled = self.pool(feats)
        vec = self.feat(pooled)  # shape [B, hidden]
        logits = self.cls_head(vec)
        hb_gdl = self.reg_head(vec).squeeze(1)  # regression in g/dL
        return logits, hb_gdl

    def grad_cam(self, class_idx: int = None) -> np.ndarray:
        # Build CAM from cached _acts (B,C,H,W) and _grads (B,C,H,W) after backward()
        if self._acts is None or self._grads is None:
            return None
        acts = self._acts[0]   # C,H,W
        grads = self._grads[0] # C,H,W

        weights = grads.mean(dim=(1, 2))  # C
        cam = torch.zeros_like(acts[0])
        for c, w in enumerate(weights):
            cam += w * acts[c]
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        cam_np = cam.cpu().numpy()
        return cam_np

# -----------------------------
# Load model
# -----------------------------

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pt")
device = get_device()

model = MultiHeadMobileNet(num_classes=3).to(device)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=device)
    # Allow either full state dict or wrapped dict
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
else:
    # Model weights missing; app will still load but warn in UI
    pass

# -----------------------------
# Inference + Grad-CAM
# -----------------------------

def predict(image: Image.Image):
    if image is None:
        return "Please upload an image.", None, None

    x = pil_to_tensor(image).to(device)
    with torch.no_grad():
        logits, hb_pred = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        hb_pred = float(hb_pred.cpu().numpy().item())

    # Determine risk label from classification OR regression
    cls_idx = int(np.argmax(probs))
    classes = ["Anemic", "Borderline", "Normal"]
    cls_label = classes[cls_idx]

    # Combine: use regression estimate (g/dL) to compute risk
    hbdl = hb_pred
    risk = risk_from_hbdl(hbdl)

    # Grad-CAM (use predicted class)
    # Backprop wrt predicted class score
    image_for_cam = pil_to_tensor(image).to(device).requires_grad_(True)
    logits_cam, _ = model(image_for_cam)
    score = logits_cam[0, cls_idx]
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)
    cam = model.grad_cam(class_idx=cls_idx)

    # Build overlay image if CAM is available
    cam_img = None
    try:
        if cam is not None:
            import cv2
            # Prepare base image
            img_np = np.array(image.convert("RGB"))
            H, W = img_np.shape[:2]
            cam_resized = cv2.resize(cam, (W, H))
            heatmap = (255 * plt_colormap(cam_resized)).astype(np.uint8)
            overlay = (0.6 * img_np + 0.4 * heatmap).astype(np.uint8)
            cam_img = Image.fromarray(overlay)
    except Exception:
        cam_img = None

    # Build a readable summary
    summary = f"Predicted Hb: {hbdl:.1f} g/dL\n" \
              f"Classification: {cls_label} (p={probs[cls_idx]:.2f})\n" \
              f"Anemia Risk: {risk}"

    return summary, hbdl, cam_img

def plt_colormap(cam01: np.ndarray) -> np.ndarray:
    # Jet colormap without importing matplotlib (build small LUT)
    # cam01 expected in [0,1]
    try:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet')
        colored = cmap(cam01)[:, :, :3]  # drop alpha
        return colored
    except Exception:
        # Fallback simple grayscale -> RGB
        c = np.clip(cam01, 0, 1)
        return np.stack([c, c*0.5, 1-c], axis=-1)

with gr.Blocks() as demo:
    gr.Markdown("# NutriScan · Non‑invasive Anemia Screening")
    gr.Markdown(
        "Upload a hand or fingernail photo. The model estimates hemoglobin (g/dL), "
        "predicts risk category, and shows a Grad‑CAM explanation."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload Image", sources=["upload", "clipboard", "webcam"])
    with gr.Row():
        out_text = gr.Textbox(label="Prediction Summary")
    with gr.Row():
        out_hb = gr.Number(label="Estimated Hemoglobin (g/dL)")
    with gr.Row():
        out_cam = gr.Image(type="pil", label="Grad‑CAM")

    btn = gr.Button("Analyze")
    btn.click(fn=predict, inputs=[inp], outputs=[out_text, out_hb, out_cam])

    if not os.path.exists(MODEL_PATH):
        gr.Markdown("> **Note:** `model.pt` was not found. "
                    "Please upload your trained weights to the Space repository root or set `MODEL_PATH`.")

if __name__ == "__main__":
    demo.launch()

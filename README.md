# ⚡ Neural Architecture Search As A Service

<div align="center">

![NAS Banner](https://img.shields.io/badge/Neural%20Architecture%20Search-As%20A%20Service-6366f1?style=for-the-badge&logo=pytorch&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-4ade80?style=flat-square)](LICENSE)
[![CPU Only](https://img.shields.io/badge/CPU-Only%20%E2%80%93%20No%20GPU%20Needed-38bdf8?style=flat-square)](https://github.com/sayansasmal/Lightweight-NAS-Engine)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/cloud)

**Automatically discover the best deep learning architecture for your image classification task — no GPU, no coding, no expertise required.**

[🚀 Live Demo](#-deployment) • [📖 Documentation](#-how-it-works) • [⚡ Quick Start](#-quick-start) • [🔬 Search Space](#-search-space)

</div>

---

## 📌 Overview

**NAS As A Service** is a lightweight, beginner-friendly Neural Architecture Search framework built for image classification. It runs entirely on a standard CPU laptop — no cloud, no GPU, no complex setup.

Upload your image dataset as a ZIP file, configure the search, and the engine automatically:
1. Evaluates multiple candidate architectures from both a **preset pool** and a **configurable custom search space**
2. Selects the best-performing model by validation accuracy
3. Fine-tunes it with full training
4. Serves it for single-image inference — all through a clean web interface

> **Final Year B.Tech Project** — Department of Computer Science and Engineering  
> College of Engineering & Management, Kolaghat (Affiliated to MAKAUT, WB)

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Unified NAS Engine** | Evaluates preset models + configurable candidates in one pipeline |
| 🧱 **3 Block Types** | PlainBlock, ResidualBlock, BottleneckBlock — 81 configurable combinations |
| 🎯 **Auto Model Selection** | Best architecture picked automatically by validation accuracy |
| ⏹ **Early Stopping** | Patience-based stopping with best-weight restoration per trial |
| 📊 **Live Terminal Log** | Real-time per-epoch metrics streamed to a styled terminal UI |
| 🌑 **Dark Dashboard UI** | Professional Space Grotesk + Syne typefaces, glassmorphism cards |
| 🩺 **Medical Imaging Ready** | Tested on ISIC skin lesion dataset (basal cell carcinoma, nevus) |
| 💾 **Model Export** | Download trained `.pt` model + `meta.json` for repeated inference |
| 🖼️ **Inference Page** | Upload any image, get prediction + confidence + probability chart |
| 🖥️ **CPU-Only** | Runs on a 4 GB RAM laptop with no GPU whatsoever |

---

## 🔬 Search Space

### Pool 1 — Preset Architectures

| Model | Parameters | Pretrained |
|---|---|---|
| TinyCNN | ~100 K | ✗ |
| SmallCNN | ~300–500 K | ✗ |
| ResNet18 | 11.2 M | ✓ ImageNet |
| MobileNetV2 | 3.4 M | ✓ ImageNet |
| EfficientNetB0 | 5.3 M | ✓ ImageNet |

### Pool 2 — Configurable Search Space

The NAS engine dynamically builds CNN architectures by combining values across four dimensions:

| Dimension | Choices | Description |
|---|---|---|
| **Depth** | 2, 3, 4 | Number of conv blocks stacked. Channels double each block. |
| **Base Channels** | 16, 32, 64 | Channels in the first block. |
| **Block Type** | plain, residual, bottleneck | Architectural style of each block. |
| **Dropout** | 0.0, 0.2, 0.5 | Dropout rate in classifier head. |

> 3 × 3 × 3 × 3 = **81 unique configurable architectures**

### Block Types Explained

```
PlainBlock      → Conv2d → BatchNorm → ReLU → MaxPool(2)
                  Simple, fast, minimal parameters. VGG-style.

ResidualBlock   → [Conv → BN → ReLU → Conv → BN] + skip connection + MaxPool(2)
                  output = F(x) + shortcut(x)
                  Solves vanishing gradients in deep nets. ResNet-inspired.

BottleneckBlock → 1×1 squeeze → 3×3 conv → 1×1 expand → MaxPool(2)
                  Parameter-efficient spatial processing in reduced channel space.
                  MobileNet/EfficientNet-inspired.
```

---

## 🏗️ Project Architecture

```
NASaaS System
│
├── User Interface Layer (Streamlit)
│   ├── 🏠  Home         — Project overview, block type cards, hero section
│   ├── 🔬  Train & Search — Full NAS workflow (upload → search → train → eval)
│   └── 🔍  Inference    — Single-image classification with confidence scores
│
├── Dataset Manager
│   ├── ZIP upload & extraction
│   ├── Auto class detection from folder names
│   ├── Stratified train/val split
│   └── Image transforms (resize, normalise, augment)
│
├── NAS Engine  ←── core contribution
│   ├── Pool 1: Preset architectures
│   ├── Pool 2: Configurable space (81 configs)
│   ├── Early stopping per trial
│   ├── Parameter budget filter
│   └── Live callback-based logging
│
├── Training Engine
│   ├── Full fine-tuning of best model
│   ├── Adam optimiser + CrossEntropyLoss
│   └── Best-weight checkpoint restoration
│
├── Evaluation Module
│   ├── Confusion matrix (seaborn dark heatmap)
│   ├── Classification report (precision, recall, F1)
│   └── Best model training curves only
│
└── Inference Module
    ├── Load best_model.pt + meta.json
    ├── Preprocess uploaded image
    └── Predict class + confidence + probability chart
```

---

## 📁 Project Structure

```
Lightweight-NAS-Engine/
│
├── app.py                  # Streamlit UI — page routing, visualisation, callbacks
├── train.py                # NAS engine, training loop, early stopping, callbacks
├── models.py               # All model definitions (preset + configurable)
├── search_space.py         # config_to_name(), estimate_params()
├── utils.py                # ImageFolderCSV dataset class, default_transforms()
├── requirements.txt        # Pinned Python dependencies
│
├── .streamlit/
│   └── config.toml         # maxUploadSize = 2048 MB
│
└── outputs/                # Created at runtime
    ├── best_model.pt        # Saved model weights
    └── meta.json            # Class names, image size, best config
```

---

## ⚡ Quick Start

### Prerequisites

- Python 3.x
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sayansasmal/Lightweight-NAS-Engine.git
cd Lightweight-NAS-Engine

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📊 How It Works

### Step 1 — Upload Dataset

Upload a ZIP file with images organised in **folder-per-class** format:

```
dataset.zip
├── cats/
│   ├── img1.jpg
│   └── img2.jpg
└── dogs/
    ├── img3.jpg
    └── img4.jpg
```

A single optional wrapper folder is also supported:
```
dataset.zip
└── animals/
    ├── cats/
    └── dogs/
```

Folder names become class labels automatically — no CSV or annotation file needed.

### Step 2 — Configure NAS Search

- Select which preset architectures to include in Pool 1
- Choose dimensions for the configurable search space (Pool 2)
- Set a parameter budget limit to skip oversized configs
- Choose between exhaustive search or random sampling

### Step 3 — Launch

Click **⚡ Launch NAS Search + Training**. The engine:

1. Trains each preset candidate for `N` epochs (with early stopping)
2. Trains each configurable candidate for `N` epochs (with early stopping)
3. Picks the single best model across both pools
4. Displays a live terminal log and progress bar throughout
5. Fine-tunes the winner for the configured number of additional epochs
6. Shows training curves, confusion matrix, and classification report

### Step 4 — Inference

Navigate to **🔍 Inference** in the sidebar, upload any image, and get:
- Predicted class name
- Confidence score (colour-coded green/yellow/red)
- Full probability bar chart for all classes
- Session prediction history

---

## 📋 Requirements

```
streamlit==1.34.0
torch==2.2.0
torchvision==0.18.0
pillow==10.0.0
matplotlib==3.8.0
scikit-learn==1.4.0
tqdm==4.66.1
albumentations==1.3.0
pandas==2.2.0
seaborn
numpy
```

**Hardware:**
- CPU only (no GPU required)
- Minimum 4 GB RAM (8 GB recommended)
- 5 GB free disk space

---

## 🧪 Testing with a Dummy Dataset

No dataset? Create a quick test one:

```python
# make_test_zip.py
import os, zipfile
from PIL import Image

for cls in ["cats", "dogs"]:
    os.makedirs(f"test_data/{cls}", exist_ok=True)

for i in range(15):
    Image.new("RGB", (128, 128), color=(i*15, 100, 200)).save(f"test_data/cats/cat_{i}.jpg")
    Image.new("RGB", (128, 128), color=(200, i*15, 100)).save(f"test_data/dogs/dog_{i}.jpg")

with zipfile.ZipFile("test_dataset.zip", "w") as z:
    for cls in ["cats", "dogs"]:
        for f in os.listdir(f"test_data/{cls}"):
            z.write(f"test_data/{cls}/{f}", f"{cls}/{f}")

print("test_dataset.zip ready")
```

Then upload `test_dataset.zip` in the app. Use `tiny_cnn` with 1–2 epochs — search completes in under 60 seconds.

---

## 🚀 Deployment

### Streamlit Cloud (recommended)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — your app gets a public URL like `your-app.streamlit.app`

Every push to `master` triggers an automatic redeploy.

### Local Network Sharing

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Other devices on the same network can access it at `http://<your-ip>:8501`.

---

## 📈 Sample Results

Results on ISIC skin lesion dataset (basal_cell_carcinoma vs nevus):

| Architecture | Type | Val Accuracy | Parameters |
|---|---|---|---|
| tiny_cnn | Preset | 0.444 | 23,938 |
| small_cnn | Preset | 0.778 | 241,794 |
| resnet18 | Preset | 0.667 | 11,177,538 |
| **mobilenet_v2** | **Preset** | **1.000** | **2,226,434** |
| efficientnet_b0 | Preset | 0.667 | 4,010,110 |
| d2_c32_residual_dr0.0 | Configurable | 0.667 | 148,770 |

**Best model selected:** `mobilenet_v2`  
**Final validation accuracy:** 1.000  
**Average inference time (CPU):** 0.2–0.5 seconds per image

---

## 🗺️ Roadmap

- [x] Preset model NAS (5 architectures)
- [x] Configurable search space (81 combinations)
- [x] Unified search combining both pools
- [x] Early stopping with patience
- [x] Live terminal logging with callbacks
- [x] Dark professional UI (Space Grotesk + Syne)
- [x] Streamlit Cloud deployment
- [ ] Proxy task scoring (zero-cost NAS)
- [ ] Grad-CAM visualisation on Inference page
- [ ] ONNX / TorchScript model export
- [ ] Learning rate scheduler support (cosine annealing)
- [ ] GPU acceleration (optional CUDA mode)
- [ ] Evolutionary / differentiable NAS strategy
- [ ] Experiment history persistence (SQLite)
- [ ] FastAPI backend for cloud-scale deployment

---

## 👥 Team

| Name | University Roll | College Roll |
|---|---|---|
| Sayan Sasmal | 10700122176 | CSE/22/163 |
| Subhamoy Maity | 10700122009 | CSE/22/029 |
| Deep Dolai | 10700122072 | CSE/22/153 |
| Sayan Patra | 10700122076 | CSE/22/154 |

**Supervisor:** Prof. Bidisha Maiti, Assistant Professor, Dept. of CSE  
**Institution:** College of Engineering & Management, Kolaghat (MAKAUT, WB)

---

## 📚 References

1. Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," IEEE, 1998.
2. A. Krizhevsky et al., "ImageNet Classification with Deep CNNs," NIPS, 2012.
3. K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
4. M. Tan & Q. V. Le, "EfficientNet: Rethinking Model Scaling for CNNs," ICML, 2019.
5. B. Zoph & Q. V. Le, "Neural Architecture Search with Reinforcement Learning," ICLR, 2017.
6. H. Liu et al., "DARTS: Differentiable Architecture Search," ICLR, 2019.
7. A. Howard et al., "Searching for MobileNetV2," ICCV, 2019.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ⚡ by the NAS As A Service Team · CEMK · MAKAUT · 2026

</div>

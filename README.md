# IU-PSeg

IU-PSeg: Identifiability and Uncertainty-Aware 3D Ovarian Tumor Segmentation in Ultrasound Imaging

## 🔬 Overview
This repository provides the official implementation of IU-PSeg, a physics-guided framework for 3D ovarian tumor segmentation that jointly models:
- Tumor probability
- Predictive uncertainty (epistemic & aleatoric)
- Identifiability

## 🧠 Key Features
- 3D ultrasound segmentation
- Uncertainty estimation (MC Dropout)
- Identifiability modeling
- Grad-CAM visualization
- Composite loss function

## 📊 Dataset
We use the MMOTU (OTU-3D) dataset.

## ⚙️ Installation
```bash
pip install -r requirements.txt

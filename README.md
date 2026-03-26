# IU-PSeg

**IU-PSeg: Identifiability and Uncertainty-Aware 3D Ovarian Tumor Segmentation in Ultrasound Imaging**

---

## 🔬 Overview

IU-PSeg is a physics-guided deep learning framework for **3D ovarian tumor segmentation in ultrasound imaging**, designed to improve both **accuracy** and **reliability**.

Unlike conventional segmentation models, IU-PSeg jointly models:
- Tumor probability
- Predictive uncertainty (epistemic & aleatoric)
- Identifiability (recoverability of tumor regions)

This enables the model to distinguish between:
- Reliable regions
- Ambiguous regions
- Non-identifiable regions

---

## 🧠 Key Contributions

- ✔ Identifiability-aware segmentation framework  
- ✔ Joint modeling of uncertainty (epistemic + aleatoric)  
- ✔ Physics-guided inverse problem formulation  
- ✔ Composite loss function for reliability-aware learning  
- ✔ Grad-CAM-based interpretability analysis  
- ✔ Pareto-efficient model (FLOPs vs Params vs Dice)  

---

## 📊 Results

IU-PSeg achieves state-of-the-art performance on 3D ultrasound data:

- **Dice:** 0.8902 ± 0.0115  
- **IoU:** 0.8319  
- **Precision:** 0.9076  
- **Recall:** 0.8991  
- **AUC:** 0.9934  

Statistically significant improvements over competing methods (**p < 0.05**).

---

## 📁 Repository Structure

```text
IU_PSeg/
│
├── models/                 # Core model architecture
│   └── IU_PSeg.py
│
├── scripts/                # Training, inference, and visualization
│   ├── evaluate_iupseg.py
│   ├── final_inference_tta.py
│   ├── final_inference_tta_full.py
│   ├── final_all_in_one_figure.py
│   ├── best_case_figure.py
│   ├── generate_figures.py
│   ├── Combined_Loss_Graph.py
│   ├── Training_Validation_Loss_Graph.py
│   ├── Pareto_optimal_models.py
│   ├── FLOPs_Params_Dice.py
│   └── 3D_ANALYSIS_(FLOPs_vs_Params_vs_Dice).py
│
├── results/
│   ├── figures/            # Generated figures
│   └── tables/             # CSV tables and analysis results
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore

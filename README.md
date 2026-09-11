# IU-PSeg

## Identifiability- and Uncertainty-Aware Probabilistic Segmentation for Reliable Ovarian Ultrasound Imaging

Official implementation of **IU-PSeg**, a physics-guided uncertainty-aware segmentation framework for reliable ovarian ultrasound analysis.

IU-PSeg addresses the challenges of ultrasound segmentation caused by speckle noise, attenuation, acoustic shadowing, anisotropic resolution, and partial-volume effects by integrating acquisition-informed perturbation modeling, uncertainty estimation, identifiability learning, boundary refinement, and reliability-aware inference.

---

# Framework Overview

IU-PSeg consists of:

- Acquisition-Informed Acoustic Perturbation Model
- Multi-Scale Acoustic Reliability Encoder (MARE)
- Reliability-Aware Skip Fusion (RASF)
- Probabilistic segmentation and uncertainty decomposition
- Perturbation-derived identifiability modeling
- Identifiability-Guided Boundary Refinement (IGBR)
- Reliability-aware inference and expert-review prediction

The framework provides:

- Tumor probability estimation
- Aleatoric uncertainty
- Epistemic uncertainty
- Identifiability maps
- Boundary response maps
- Voxel-wise reliability
- Case-level reliability assessment

---

# Repository Structure



---

# Installation

Clone the repository:

```bash
git clone https://github.com/myyaqubpython/IU_PSeg.git

cd IU_PSeg

pip install -r requirements.txt


Datasets

IU-PSeg was evaluated on:

MR-3DUS

A volumetric ovarian ultrasound dataset used for 3D segmentation evaluation.

CEUS-OLSeg

A contrast-enhanced ultrasound lesion segmentation dataset used for additional validation.

Due to clinical privacy restrictions, raw ultrasound images and annotations are not publicly released.


Training

Configure training parameters:

configs/

Train IU-PSeg:

python train/train.py \
--config configs/default.yaml

Inference

Run inference:

python inference/test.py \
--checkpoint checkpoints/model.pth

Evaluation

Generate quantitative results:

python evaluation/evaluate.py

Metrics include:

Dice coefficient
IoU
Precision
Recall
HD95
ASD
AUC
Reliability analysis

Visualization

Generate:

Segmentation masks
Probability maps
Uncertainty maps
Identifiability maps
Reliability maps
python visualization/generate_maps.py

Pretrained Models

Pretrained weights will be released after publication.


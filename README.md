<div align="center">
<img width="600" height="289" alt="Basmallah-4-White-940x453" src="https://github.com/user-attachments/assets/d3937692-adaa-4eb2-9998-c55c384c9a81" />
</div>

---

# Task Optimization Drives Statistical Decorrelation

## An Empirical Study of Integration Dynamics in Feed-forward and Recurrent Neural Networks

<p align="center">
  <img width="3000" height="1800" alt="Figure_1" src="https://github.com/user-attachments/assets/224d98b0-2c1e-4858-94ac-1e912bdc7d7c" />
</p>

## Overview

This repository contains the official implementation and experimental results for the research paper:

> **Task Optimization Drives Statistical Decorrelation: An Empirical Study of Integration Dynamics in Feed-forward and Recurrent Neural Networks**

The project investigates the relationship between:

* **Task Optimization** (classification accuracy)
* **Statistical Information Integration**
* **Representation Decorrelation**
* **Neural Specialization during Learning**

using Feed-forward Neural Networks (MLPs) and Recurrent Neural Networks (RNNs).

---

# Motivation

Integrated Information Theory (IIT) proposes that consciousness is related to a system's capacity for information integration (Φ).

Modern Deep Learning systems, however, are optimized primarily for task performance and generalization. Representation learning theories such as the Information Bottleneck principle suggest that successful optimization progressively compresses and decorrelates internal representations.

This project explores the following central question:

> **Does increasing task intelligence necessarily preserve information integration?**

Our empirical results suggest a strong inverse relationship:

* Higher accuracy → lower redundancy
* Higher specialization → lower integration
* Optimization progressively decorrelates hidden representations

---

# Core Hypothesis

We hypothesize that:

> Standard gradient-based optimization drives neural networks toward increasingly decorrelated internal representations, reducing statistical integration as performance improves.

This creates a potential trade-off between:

| Intelligence                | Integration                   |
| --------------------------- | ----------------------------- |
| Efficient task solving      | Holistic information coupling |
| Specialized representations | Redundant integrated states   |
| Error minimization          | Statistical dependency        |

---

# Main Contributions

## 1. Empirical Analysis of Integration Dynamics

We analyze how Total Correlation (TC) evolves during neural network training.

## 2. MLP vs RNN Comparison

We compare feed-forward and recurrent architectures under identical optimization conditions.

## 3. Statistical Proxy for Integration

Instead of computing the computationally intractable causal Φ from IIT, we use:

* Gaussian Total Correlation (TC)

as a tractable proxy for multivariate statistical dependency.

## 4. Evidence of Optimization-Driven Decorrelation

Our experiments consistently demonstrate:

* increasing accuracy
* decreasing TC
* progressive specialization
* collapse of redundancy during learning

---

# Experimental Setup

## Architectures

### Feed-forward Network (MLP)

* Sparse Linear Layers
* Tanh activations
* Optional residual connections

### Recurrent Neural Network (RNN)

* Temporal feedback loops
* Hidden-state recurrence
* Multi-step temporal integration

---

# Datasets

Experiments were performed on:

* XOR
* Two Moons
* Spiral Dataset

These tasks were selected because they:

* require non-linear decision boundaries
* expose internal representation dynamics
* allow controlled analysis of integration collapse

---

# Information Integration Metric

We estimate integration using Gaussian Total Correlation:

[
TC(X) = \frac{1}{2}\left( \log |D| - \log |\Sigma| \right)
]

Where:

* (\Sigma) is the covariance matrix
* (D) is the diagonal covariance matrix

Higher TC values indicate:

* stronger statistical dependency
* greater redundancy
* higher integration among hidden units

---

# Key Findings

## Feed-forward Networks (MLP)

| Phase          | Accuracy | TC         |
| -------------- | -------- | ---------- |
| Early Training | ~55%     | High (~35) |
| Final Training | ~88%     | Low (~16)  |

Observation:

> Optimization progressively reduces information redundancy.

---

## Recurrent Networks (RNN)

| Phase          | Accuracy | TC              |
| -------------- | -------- | --------------- |
| Early Training | ~69%     | Very High (~38) |
| Final Training | ~93%     | Reduced (~20)   |

Observation:

> Recurrent feedback preserves integration longer, but optimization still drives eventual decorrelation.

---

# Interpretation

The experiments suggest the emergence of a fundamental phenomenon:

## Specialization vs Integration

During training:

* neurons become increasingly specialized
* redundant representations collapse
* hidden units decorrelate
* integration decreases

This implies that:

> high task performance does not necessarily imply high integration.

Under IIT-inspired interpretations:

> current neural architectures may optimize for intelligence while simultaneously suppressing integrated states.

---

# Repository Structure

```text
integration-dynamics-neural-networks/
│
├── README.md
├── requirements.txt
│
├── code/
│   ├── integration_proxy_tc.py
│   └── plot_results.py
│
├── results/
│   ├── results.csv
│   └── results_enhanced.csv
│
├── figures/
│   ├── Figure_1.png
│   ├── Figure_2.png
│   ├── Figure_3.png
│   ├── Figure_4.png
│   ├── Figure_5.png
│   └── Figure_6.png
│
├── paper/
│   └── integration_dynamics_iit.pdf
│
└── supplementary/
    └── iit_binary_causal_demo.py
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/SAYED-ZALABIYA/integration-dynamics-neural-networks.git
cd integration-dynamics-neural-networks
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Main dependencies:

* Python 3.10+
* PyTorch
* NumPy
* Matplotlib
* Pandas
* Seaborn

Optional:

* PyPhi (for toy Φ demonstrations)

---

# Running Experiments

## Basic XOR Experiment

```bash
python code/integration_proxy_tc.py --dataset xor
```

---

## Two Moons Experiment

```bash
python code/integration_proxy_tc.py --dataset moons
```

---

## Full Grid Search

```bash
python code/integration_proxy_tc.py --grid 1
```

---

## Generate Visualizations

```bash
python code/integration_proxy_tc.py --visualize --viz_layers --viz_summary
```

---

# Main Figures

## Figure 1 — Accuracy vs Total Correlation
<div align="center">
<img width="3000" height="1800" alt="Figure_1" src="https://github.com/user-attachments/assets/7268482e-0abf-4580-85e9-318f1f73319b" />
</div>
Demonstrates the inverse relationship between optimization and integration.

---

## Figure 2 — Layer-wise Decorrelation
<div align="center">
<img width="3000" height="1800" alt="Figure_2" src="https://github.com/user-attachments/assets/c68f1ec2-2897-4744-a4c2-ecfc3b50b8f3" />
</div>
Shows progressive specialization across hidden layers.

---

## Figure 3 — MLP vs RNN Integration Dynamics
<div align="center">
<img width="3000" height="1800" alt="Figure_3" src="https://github.com/user-attachments/assets/7c7e0d55-c0b1-4dea-b9dd-6c4cb54fd7d1" />
</div>
Compares integration decay across architectures.

---

## Figure 4 — Temporal Integration in RNNs
<div align="center">
<img width="3000" height="1800" alt="Figure_4" src="https://github.com/user-attachments/assets/f94fb605-1a36-4e8a-8dcb-d85163e27556" />
</div>
Illustrates persistence of integration through recurrent feedback.

---

## Figure 5 — Accuracy vs TC Correlation
<div align="center">
<img width="3000" height="1800" alt="Figure_5" src="https://github.com/user-attachments/assets/2e6a6e10-865b-402f-a1d0-503320509f9b" />
</div>
Reveals strong negative correlation between performance and integration.

---

## Figure 6 — Summary of Integration Collapse
<div align="center">
<img width="3000" height="1800" alt="Figure_6" src="https://github.com/user-attachments/assets/61f06800-bdc6-4309-9f75-1e2936c510dd" />
</div>
Highlights global reduction in TC between initialization and convergence.

---

# Important Scientific Clarification

This project does **NOT** claim:

* measurement of true consciousness
* computation of exact IIT Φ for large neural systems
* proof of conscious behavior in artificial neural networks

Instead, this work investigates:

> statistical integration dynamics during optimization

using Total Correlation (TC) as a tractable approximation of multivariate dependency.

---

# Future Directions

Potential extensions include:

* Transformer integration analysis
* Attention-based integration mechanisms
* Causal Φ approximations
* Integration-preserving regularization
* Scaling to large language models
* Effective Information (EI) interventions

---

# Citation

If you use this repository or build upon this work, please cite:

```bibtex
@article{mohammed2026integration,
  title={Task Optimization Drives Statistical Decorrelation: An Empirical Study of Integration Dynamics in Feed-forward and Recurrent Neural Networks},
  author={Mohammed, ElSayed A.},
  year={2026},
  journal={arXiv preprint}
}
```

---

# Author

**ElSayed A. Mohammed**

Independent Researcher

Egypt

---

# License

This project is released under the MIT License.

---
<div align="center">
<img width="736" height="840" alt="WhatsApp Image 2026-04-22 at 23 49 24" src="https://github.com/user-attachments/assets/00eb0078-9039-409e-bd16-c4a8e6cd314c" />
</div>

---
<div align="center">
a7la msa 3ly altrmsa :)
</div>

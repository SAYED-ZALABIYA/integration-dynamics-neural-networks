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



















# Exploring Integration in Artificial Neural Systems  
### Statistical Proxies vs Causal IIT Measures

This repository contains a research-oriented prototype that explores **information integration**
in artificial systems by connecting:

- **Statistical integration** inside neural networks  
- **Causal integration** as defined by **Integrated Information Theory (IIT)**

The project does **not** attempt to claim consciousness in AI.  
Its goal is to study **measurable bridges** between representation learning, information structure, and causal mechanisms.

---

## Conceptual Overview

Integrated Information Theory (IIT) defines **Φ (Phi)** as a measure of how much a system is
causally irreducible.

However, computing Φ directly for neural networks is intractable.

This project explores a two-track approach:

| Track | Level | What is Measured |
|-----|------|----------------|
| Phase 1 | Statistical | Total Correlation (TC) in neural activations |
| Phase 2 | Causal | True Φ from explicit Transition Probability Matrices |

The central research question:

> **Can statistical integration trends inside neural networks approximate or predict causal Φ?**

---

## Repository Structure

```text
src/
 ├── integration_proxy_tc.py    # Phase 1: TC in neural networks
 └── iit_causal_phi_tpm.py      # Phase 2: Φ from binary causal systems

data/
 ├── results_enhanced.csv
 └── epoch_data.json

assets/
 └── figures/                  # Generated visualizations
```

## Phase 1 — Statistical Integration (Proxy)

File: src/integration_proxy_tc.py

### A lightweight Tiny MLP trained on simple datasets:

- XOR

- Two Moons

- Spiral

### What is tracked

- Total Correlation (TC) per layer

- TC Sum across layers

- Accuracy vs training epochs

- Effects of connectivity density and residual links

### Purpose

TC acts as a statistical proxy for how integrated learned representations become during training.

⚠️ Any Φ value inside this phase (if present) is a toy demonstration only,
not causal IIT Φ.


### Phase 2 — Causal Integration (IIT Ground Truth)

File: src/iit_causal_phi_tpm.py

This phase isolates causality from learning.

### What is implemented

- Explicit binary systems (3–5 nodes)

- Deterministic logical architectures (AND, OR, XOR, NAND chains)

- Construction of full Transition Probability Matrices (TPMs)

- Exact computation of Φ using PyPhi

### Purpose

Provides a mechanistic ground truth for causal integration, fully aligned with IIT.

### Example Results (Preliminary)

- In initial runs, TC tends to decrease and compress as accuracy increases.

- This suggests a possible relationship between learning dynamics and integration structure.

   - Results are exploratory and single-run at this stage.
   - Statistical significance is not claimed yet.
 
## Installation

```text
pip install -r requirements.txt
```

### Running the Experiments

### Phase 1 — Neural Integration

```text
python src/integration_proxy_tc.py --dataset xor --epochs 40 --visualize
```
### Phase 2 — Causal Φ

```text
python src/iit_causal_phi_tpm.py --preset xor_nand_5
python src/iit_causal_phi_tpm.py --preset and_or_4 --state 1010
```

## Research Direction (Next Steps)

- Binarization of neural activations

- Reduced-state causal abstraction of trained networks

- Testing whether TC trends predict Φ under controlled reductions

- Multi-run statistical validation

## Key Takeaway

### This project demonstrates a methodological bridge between:

- Representation learning

- Information theory

- Causal structure

### It is a step toward quantifying integration in artificial systems, not defining consciousness.

## Author

### Elsayed Nassar

## لا تنسونا من صالح دعائكم 

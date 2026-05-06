<div align="center">
<img width="600" height="289" alt="Basmallah-4-White-940x453" src="https://github.com/user-attachments/assets/d3937692-adaa-4eb2-9998-c55c384c9a81" />
</div>


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

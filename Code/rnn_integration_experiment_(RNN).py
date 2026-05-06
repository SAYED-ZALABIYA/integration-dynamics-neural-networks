import math, random, argparse, csv, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import warnings

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


try: import seaborn as sns; HAS_SNS = True
except: HAS_SNS = False
try: import pandas as pd; HAS_PD = True
except: HAS_PD = False
try: import pyphi; PYPHI_AVAILABLE = True
except: PYPHI_AVAILABLE = False

# ================= Utils =================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_xor(n=1024, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.int64)
    X = X + rng.normal(0, noise, X.shape)
    X = (X - X.mean(0)) / X.std(0)
    return X.astype(np.float32), y

def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-6)

def stable_cov(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, X.shape[0]-1)
    return C + np.eye(C.shape[0]) * eps

def gaussian_total_correlation(X: np.ndarray, eps: float = 1e-6) -> float:
    C = stable_cov(X, eps)
    D = np.diag(np.diag(C))
    s1, ld1 = np.linalg.slogdet(D); s2, ld2 = np.linalg.slogdet(C)
    return float(max(0.0, 0.5*(ld1 - ld2)))

# ================= RNN Model (The Feedback Loop) =================
"""
Minimal recurrent neural network used to analyze
temporal statistical integration dynamics.
"""
class TinyRNN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=8, out_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.RNNCell(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, steps=3, collect=False):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        all_acts = []
        for t in range(steps):
            h = self.rnn_cell(x, h)
            h = torch.tanh(h)
            if collect: all_acts.append(h.detach().cpu().numpy())
        logits = self.classifier(h)
        return logits, all_acts

# ================= IIT Proxy =================
"""
Toy approximation of Φ using PyPhi.

This is NOT a valid causal IIT Φ computation for large-scale neural systems.
Used only for illustrative low-dimensional demonstrations.
"""
def compute_phi_toy(acts: np.ndarray, threshold: float = 0.0) -> float:
    if not PYPHI_AVAILABLE: return 0.0
    try:
        bin_states = (acts > threshold).astype(int)
        
        max_u = min(6, bin_states.shape[1])
        bin_states = bin_states[:, :max_u]

        
        num_states = 2**max_u
        tpm = np.zeros((num_states, max_u))
        rng = np.random.RandomState(42)
        tpm = rng.rand(num_states, max_u)

        net = pyphi.Network(tpm)
        state = tuple((bin_states.mean(0) > 0.5).astype(int))
        sub = pyphi.Subsystem(net, state, range(max_u))
        return float(pyphi.compute.phi(sub))
    except: return 0.0

# ================= Execution =================
@dataclass
class RunConfig:
    dataset: str = "xor"
    n_samples: int = 1024
    seed: int = 0
    hidden_dim: int = 8
    steps: int = 3 
    lr: float = 0.01
    epochs: int = 40

def run_once(cfg: RunConfig):
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = make_xor(cfg.n_samples, seed=cfg.seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=64, shuffle=True)

    model = TinyRNN(2, cfg.hidden_dim, 2).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    logs = []
    for ep in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, _ = model(xb, steps=cfg.steps)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # Evaluation & IIT Metrics
        model.eval()
        with torch.no_grad():
            full_x = torch.from_numpy(X).to(dev)
            logits, acts_steps = model(full_x, steps=cfg.steps, collect=True)
            acc = (logits.argmax(1) == torch.from_numpy(y).to(dev)).float().mean().item()

            
            tc_vals = [gaussian_total_correlation(_z(step_act)) for step_act in acts_steps]
            phi_val = compute_phi_toy(acts_steps[-1]) 

            logs.append({
                "epoch": ep + 1,
                "accuracy": acc,
                "tc_sum": sum(tc_vals),
                "phi": phi_val,
                **{f"tc_step_{i+1}": v for i, v in enumerate(tc_vals)}
            })
            print(f"Epoch {ep+1}: Acc={acc:.3f}, TC={sum(tc_vals):.3f}, Phi={phi_val:.3f}")

    return logs

if __name__ == "__main__":
    config = RunConfig()
    results = run_once(config)


    with open("rnn_iit_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("Done! Results saved to rnn_iit_results.csv")
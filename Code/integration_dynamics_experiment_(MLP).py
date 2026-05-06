import math, random, argparse, csv, json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import warnings

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---- Matplotlib is required for base plots
import matplotlib.pyplot as plt

# ---- Optional deps
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False
try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False

# ---- Optional PyPhi (toy Φ demo only)
try:
    import pyphi
    PYPHI_AVAILABLE = True
except Exception:
    PYPHI_AVAILABLE = False

# ================= Utils =================
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def make_xor(n=512, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 2)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(np.int64)
    X = X + rng.normal(0, noise, X.shape)
    X = (X - X.mean(0)) / X.std(0)
    return X.astype(np.float32), y

def make_two_moons(n=1000, noise=0.15, seed=0):
    rng = np.random.RandomState(seed)
    ang = rng.rand(n) * math.pi
    r = 1.0 + rng.normal(0, noise, n)
    x1, y1 = r*np.cos(ang), r*np.sin(ang)
    x2, y2 = 1.0 - r*np.cos(ang) + 0.5, -r*np.sin(ang)
    X = np.vstack([np.stack([x1,y1],1), np.stack([x2,y2],1)]).astype(np.float32)
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    X = (X - X.mean(0)) / X.std(0)
    return X, y

def make_spiral(n=1000, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    n = n // 2
    th1 = np.sqrt(rng.rand(n)) * 2*np.pi; r1 = th1 + rng.normal(0, noise, n)
    x1, y1 = r1*np.cos(th1), r1*np.sin(th1)
    th2 = np.sqrt(rng.rand(n)) * 2*np.pi; r2 = -th2 + rng.normal(0, noise, n)
    x2, y2 = r2*np.cos(th2), r2*np.sin(th2)
    X = np.vstack([np.stack([x1,y1],1), np.stack([x2,y2],1)]).astype(np.float32)
    y = np.array([0]*n + [1]*n, dtype=np.int64)
    X = (X - X.mean(0)) / X.std(0)
    return X, y

def _z(x: np.ndarray) -> np.ndarray:
    m = x.mean(0, keepdims=True); s = x.std(0, keepdims=True) + 1e-6
    return (x - m) / s

def stable_cov(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / max(1, X.shape[0]-1)
    return C + np.eye(C.shape[0]) * eps

def gaussian_total_correlation(X: np.ndarray, eps: float = 1e-6) -> float:
    C = stable_cov(X, eps)
    D = np.diag(np.diag(C))
    s1, ld1 = np.linalg.slogdet(D); s2, ld2 = np.linalg.slogdet(C)
    if s1 <= 0 or s2 <= 0: return 0.0
    return float(max(0.0, 0.5*(ld1 - ld2)))

# ================= Model =================
class SparseLinear(nn.Module):
    def __init__(self, in_f, out_f, p_connect=1.0, seed=0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias   = nn.Parameter(torch.zeros(out_f))
        nn.init.xavier_uniform_(self.weight)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("mask", (torch.rand(out_f, in_f, generator=g) < p_connect).float())
    def forward(self, x): return F.linear(x, self.weight * self.mask, self.bias)

class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, widths: List[int] = (8,8), p_connect=1.0, residual=False, seed=0):
        super().__init__()
        dims = [in_dim] + list(widths)
        self.blocks = nn.ModuleList()
        for i in range(len(widths)):
            self.blocks.append(SparseLinear(dims[i], dims[i+1], p_connect, seed+i))
            self.blocks.append(nn.Tanh())
        self.out = SparseLinear(widths[-1], 2, 1.0, seed+999)
        self.residual = residual
    def forward(self, x, collect=False):
        acts = [] if collect else None
        h = x
        for m in self.blocks:
            if isinstance(m, SparseLinear):
                z = m(h)
                if self.residual and z.shape[-1] == h.shape[-1]: z = z + h
                h = z
            else:
                h = m(h)
                if collect: acts.append(h)
        logits = self.out(h)
        return (logits, acts) if collect else (logits, None)

# ============== Train/Eval ==============
@dataclass
class RunConfig:
    dataset: str = "xor"          # xor | moons | spiral
    n_samples: int = 1024
    noise: float = 0.1
    seed: int = 0
    widths: Tuple[int, ...] = (8, 8)
    p_connect: float = 1.0
    residual: bool = False
    batch_size: int = 128
    lr: float = 1e-2
    epochs: int = 40

def make_dataset(cfg: RunConfig):
    if cfg.dataset == "xor":
        X, y = make_xor(cfg.n_samples, cfg.noise, cfg.seed)
    elif cfg.dataset == "moons":
        X, y = make_two_moons(cfg.n_samples//2, cfg.noise, cfg.seed)
    elif cfg.dataset == "spiral":
        X, y = make_spiral(cfg.n_samples//2, cfg.noise, cfg.seed)
    else:
        raise ValueError("unknown dataset")
    n = len(X); idx = np.arange(n); np.random.RandomState(cfg.seed).shuffle(idx)
    tr = int(0.8*n); id_tr, id_te = idx[:tr], idx[tr:]
    return (torch.from_numpy(X[id_tr]), torch.from_numpy(y[id_tr])), (torch.from_numpy(X[id_te]), torch.from_numpy(y[id_te]))

def evaluate(model, loader, dev) -> float:
    model.eval(); corr=tot=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.to(dev)
            preds = model(xb)[0].argmax(1)
            corr += (preds==yb).sum().item(); tot += yb.numel()
    return corr/max(1,tot)

def compute_phi_toy(acts: np.ndarray, threshold: float = 0.5) -> float:
    """Toy Φ demo (NOT causal IIT). Disabled unless --phi_demo is set."""
    if not PYPHI_AVAILABLE: return 0.0
    try:
        bin_states = (acts > threshold).astype(int)
        max_units = 8
        if bin_states.shape[1] > max_units:
            idx = np.random.RandomState(0).choice(bin_states.shape[1], size=max_units, replace=False)
            bin_states = bin_states[:, idx]
        cov = np.cov(bin_states.T)
        tpm = 1/(1+np.exp(-cov))  # placeholder
        net = pyphi.Network(tpm)
        state = tuple((bin_states.mean(0) > 0.5).astype(int))
        sub = pyphi.Subsystem(net, state, range(len(state)))
        return float(pyphi.compute.phi(sub))
    except Exception:
        return 0.0

def run_once(cfg: RunConfig, visualize=False, viz_layers=False, phi_demo=False, act_batches: Optional[int]=None):
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (Xtr,ytr),(Xte,yte) = make_dataset(cfg)
    tr = DataLoader(TensorDataset(Xtr,ytr), batch_size=cfg.batch_size, shuffle=True)
    te = DataLoader(TensorDataset(Xte,yte), batch_size=cfg.batch_size, shuffle=False)

    net = TinyMLP(2, list(cfg.widths), cfg.p_connect, cfg.residual, cfg.seed).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    logs = []
    epoch_buf: Dict[str, List] = defaultdict(list)

    for ep in range(cfg.epochs):
        net.train()
        for xb,yb in tr:
            xb=xb.to(dev); yb=yb.to(dev)
            logits,_ = net(xb, collect=True)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        acc = evaluate(net, te, dev)

        with torch.no_grad():
            acts_lists = [[] for _ in range(len(cfg.widths))]
            seen = 0
            for xb,_ in te:
                xb = xb.to(dev)
                _, acts = net(xb, collect=True)
                for i,a in enumerate(acts):
                    acts_lists[i].append(a.cpu().numpy())
                seen += 1
                if act_batches is not None and seen >= act_batches:
                    break
            acts_all = [np.vstack(a) for a in acts_lists]
            tc_vals = [gaussian_total_correlation(_z(a)) for a in acts_all]
            tc_sum = float(sum(tc_vals))
            phi_val = compute_phi_toy(acts_all[-1]) if (phi_demo and PYPHI_AVAILABLE and len(acts_all)>0) else 0.0

        epoch_buf['epoch'].append(ep+1)
        epoch_buf['accuracy'].append(acc)
        epoch_buf['tc_sum'].append(tc_sum)
        epoch_buf['phi'].append(phi_val)
        for i,tv in enumerate(tc_vals): epoch_buf[f'tc_layer_{i+1}'].append(tv)

        logs.append({"epoch": ep+1, "acc": acc, "tc_sum": tc_sum, "phi": phi_val, **{f"tc_l{i+1}":v for i,v in enumerate(tc_vals)}})

    if visualize:
        _viz_training(epoch_buf, cfg)
    if viz_layers:
        _viz_layer_tc(epoch_buf, cfg)

    return logs, dict(epoch_buf)

# ============== Viz =================
def _viz_training(epoch_data: Dict[str, List], cfg: RunConfig):
    plt.figure(figsize=(12,5))
    ax1 = plt.gca()
    ax1.plot(epoch_data['epoch'], epoch_data['accuracy'], label="Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax2 = ax1.twinx()
    ax2.plot(epoch_data['epoch'], epoch_data['tc_sum'], '--', label="TC Sum", color='tab:red')
    ax2.set_ylabel("TC Sum")
    plt.title(f"{cfg.dataset}: Accuracy & TC over time")
    plt.tight_layout(); plt.savefig(f"training_{cfg.dataset}_s{cfg.seed}.png", dpi=300); plt.close()

def _viz_layer_tc(epoch_data: Dict[str, List], cfg: RunConfig):
    layer_keys = [k for k in epoch_data.keys() if k.startswith('tc_layer_')]
    if not layer_keys: return
    if not (HAS_SNS and HAS_PD):
        warnings.warn("Layer-wise TC plot requires seaborn and pandas. Skipping.")
        return
    epochs = epoch_data['epoch']
    rows = []
    for key in layer_keys:
        layer_num = int(key.split('_')[-1])
        for ep, tc_val in zip(epochs, epoch_data[key]):
            rows.append({'Epoch': ep, 'Layer': f'Layer {layer_num}', 'TC': tc_val})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='Epoch', y='TC', hue='Layer', marker='o')
    plt.title(f'Layer-wise Total Correlation - {cfg.dataset}')
    plt.xlabel('Epoch'); plt.ylabel('Total Correlation (TC)')
    plt.legend(title='Layers'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f'layer_tc_{cfg.dataset}_s{cfg.seed}.png', dpi=300); plt.close()

def generate_summary_visualizations(results: List[Dict[str, Any]]):
    if not (HAS_SNS and HAS_PD):
        warnings.warn("Summary visualizations require seaborn and pandas. Skipping.")
        return
    df = pd.DataFrame(results)

    # TC vs Accuracy
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='tc_sum', y='acc', hue='dataset', style='residual', s=100)
    plt.title('Total Correlation vs Accuracy Across Runs')
    plt.xlabel('TC Sum'); plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig('tc_vs_accuracy_summary.png', dpi=300, bbox_inches='tight'); plt.close()

    # Layer-wise TC distribution
    tc_cols = [c for c in df.columns if c.startswith('tc_l')]
    if tc_cols:
        tc_df = df[['dataset','residual'] + tc_cols].copy()
        tc_m = tc_df.melt(id_vars=['dataset','residual'], value_vars=tc_cols,
                          var_name='layer', value_name='tc')
        tc_m['layer'] = tc_m['layer'].str.replace('tc_l', 'Layer ', regex=False)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=tc_m, x='layer', y='tc', hue='dataset')
        plt.title('Distribution of Layer-wise TC Across Datasets')
        plt.xlabel('Layer'); plt.ylabel('Total Correlation (TC)')
        plt.xticks(rotation=45); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig('layer_tc_comparison.png', dpi=300, bbox_inches='tight'); plt.close()

# ============== Main =================
def main(argv=None):
    parser = argparse.ArgumentParser(description="IIT-inspired integration proxy (merged)")
    parser.add_argument("--dataset", choices=["xor","moons","spiral"], default="xor")
    parser.add_argument("--widths", nargs="+", type=int, default=[8,8])
    parser.add_argument("--p_connect", type=float, default=1.0)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=0)
    parser.add_argument("--fast", type=int, default=1, help="smaller grid if 1")
    parser.add_argument("--visualize", action="store_true", help="per-run Accuracy/TC plot")
    parser.add_argument("--viz_layers", action="store_true", help="per-run layer-wise TC lines")
    parser.add_argument("--viz_summary", action="store_true", help="cross-run summary figures")
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--phi_demo", action="store_true", help="enable toy Φ demo (NOT causal IIT)")
    parser.add_argument("--act_batches", type=int, default=None, help="limit batches for TC to speed up")
    args = parser.parse_args(argv)

    cfgs: List[RunConfig] = []
    if args.grid:
        if args.fast:
            seeds=[0,1]; widths=[(4,4),(8,8),(16,16)]
            pcon=[0.4,0.8,1.0]; residuals=[False,True]; dsets=[args.dataset]
        else:
            seeds=list(range(5)); widths=[(4,4),(8,8),(16,16),(32,32)]
            pcon=[0.2,0.4,0.6,0.8,1.0]; residuals=[False,True]; dsets=["xor","moons","spiral"]
        for ds in dsets:
            for s in seeds:
                for w in widths:
                    for p in pcon:
                        for r in residuals:
                            cfgs.append(RunConfig(dataset=ds, widths=w, p_connect=p, residual=r,
                                                  seed=s, epochs=args.epochs, n_samples=args.n_samples))
    else:
        cfgs.append(RunConfig(dataset=args.dataset, widths=tuple(args.widths),
                              p_connect=args.p_connect, residual=args.residual,
                              seed=args.seed, epochs=args.epochs, n_samples=args.n_samples))

    all_rows: List[Dict[str,Any]] = []
    epoch_json = []

    for i,cfg in enumerate(cfgs):
        print(f"[{i+1}/{len(cfgs)}] {cfg}")
        logs, ep_data = run_once(cfg,
                                 visualize=args.visualize,
                                 viz_layers=args.viz_layers,
                                 phi_demo=args.phi_demo,
                                 act_batches=args.act_batches)
        epoch_json.append(ep_data)
        for row in logs:
            all_rows.append({"dataset": cfg.dataset,
                             "widths": str(cfg.widths),
                             "p_connect": cfg.p_connect,
                             "residual": cfg.residual,
                             "seed": cfg.seed,
                             **row})

    # dynamic CSV fields (support variable #layers)
    max_L = 0
    for r in all_rows:
        L = sum(1 for k in r if k.startswith("tc_l"))
        max_L = max(max_L, L)
    fields = ["dataset","widths","p_connect","residual","seed","epoch","acc","tc_sum","phi"] + [f"tc_l{i+1}" for i in range(max_L)]

    with open("results_enhanced.csv","w",newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader()
        for r in all_rows:
            for i in range(max_L): r.setdefault(f"tc_l{i+1}", 0.0)
            wr.writerow(r)

    with open("epoch_data.json","w") as f:
        json.dump(epoch_json, f)

    print(f"Wrote results_enhanced.csv ({len(cfgs)} runs), epoch_data.json")

    # Optional cross-run summaries
    if args.viz_summary and len(all_rows)>0:
        generate_summary_visualizations(all_rows)

if __name__ == "__main__":
    main() # Pass an empty list to avoid unexpected arguments
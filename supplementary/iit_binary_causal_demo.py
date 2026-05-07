
import sys, types, joblib as _joblib

shim = types.ModuleType("joblib")

for name in dir(_joblib):
    setattr(shim, name, getattr(_joblib, name))

class _MemoryWrapper(_joblib.Memory):
    def __init__(self, *args, **kwargs):
        
        if "cachedir" in kwargs and "location" not in kwargs:
            kwargs["location"] = kwargs.pop("cachedir")
        super().__init__(*args, **kwargs)

shim.Memory = _MemoryWrapper
sys.modules["joblib"] = shim

import pyphi

import itertools
import argparse
import numpy as np

try:
    import pyphi
except ImportError as e:
    raise SystemExit(" PyPhi is not installed. Install it first: pip install pyphi :) ") from e



def ID(x):        return x
def NOT(x):       return 1 - x
def AND(a, b):    return a & b
def OR(a, b):     return a | b
def XOR(a, b):    return a ^ b
def NAND(a, b):   return 1 - (a & b)
def NOR(a, b):    return 1 - (a | b)


def make_network_spec(preset: str):
    preset = preset.lower()
    if preset == "and_fan_in_3":  
        labels = ["A", "B", "C"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return AND(s[0], s[1])
        funcs  = [fA, fB, fC]
        cm = np.array([
            [1,0,0],  # A depends on A
            [0,1,0],  # B depends on B
            [1,1,0],  # C depends on A,B
        ], dtype=int)

    elif preset == "xor_with_memory_3":  
        labels = ["A", "B", "C"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return XOR(s[0], s[1])
        funcs  = [fA, fB, fC]
        cm = np.array([
            [1,0,0],
            [0,1,0],
            [1,1,0],  
        ], dtype=int)

    elif preset == "and_or_4":  
        labels = ["A", "B", "C", "D"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return AND(s[0], s[1])     # AND(A,B)
        def fD(s): return OR(s[1], s[2])      # OR(B,C)
        funcs  = [fA, fB, fC, fD]
        cm = np.array([
            [1,0,0,0],  # A<-A
            [0,1,0,0],  # B<-B
            [1,1,0,0],  # C<-A,B
            [0,1,1,0],  # D<-B,C
        ], dtype=int)

    elif preset == "xor_nand_5":  
        labels = ["A", "B", "C", "D", "E"]
        def fA(s): return ID(s[0])
        def fB(s): return ID(s[1])
        def fC(s): return XOR(s[0], s[1])        # XOR(A,B)
        def fD(s): return NAND(s[1], s[2])       # NAND(B,C)
        def fE(s): return AND(s[2], s[3])        # AND(C,D)
        funcs  = [fA, fB, fC, fD, fE]
        cm = np.array([
            [1,0,0,0,0],  # A<-A
            [0,1,0,0,0],  # B<-B
            [1,1,0,0,0],  # C<-A,B
            [0,1,1,0,0],  # D<-B,C
            [0,0,1,1,0],  # E<-C,D
        ], dtype=int)
    else:
        raise ValueError("Preset unknown")

    return funcs, cm, labels



def enumerate_states(n):
    return list(itertools.product([0,1], repeat=n))

def build_tpm_state_by_node(funcs, n):
    states = enumerate_states(n)
    tpm = np.zeros((2**n, n), dtype=float)
    for idx, s in enumerate(states):
        s = tuple(int(x) for x in s)
        nxt = [f(s) for f in funcs]         
        tpm[idx, :] = np.array(nxt, dtype=float)  
    return tpm

# ---------- PyPhi ----------
def compute_phi_for_state(tpm, cm, labels, state_tuple):
    net = pyphi.Network(tpm, connectivity_matrix=cm, node_labels=labels)
    sub = pyphi.Subsystem(net, state_tuple, tuple(range(len(labels))))
    phi_value = pyphi.compute.phi(sub)
    mip = pyphi.compute.sia(sub)  
    return float(phi_value), mip


def main():
    p = argparse.ArgumentParser(description="Causal IIT demo with true binary TPM + PyPhi Φ")
    p.add_argument("--preset", choices=["and_fan_in_3","xor_with_memory_3","and_or_4","xor_nand_5"],
                   default="and_fan_in_3")
    p.add_argument("--state", type=str, default=None,
                   help="A binary state like 101 for n nodes. If not specified, Φ will be calculated for all cases.")
    p.add_argument("--show_mip", action="store_true", help="Basic IP symptoms and details.")
    args = p.parse_args()

    funcs, cm, labels = make_network_spec(args.preset)
    n = len(labels)
    tpm = build_tpm_state_by_node(funcs, n)

    if args.state is not None:
        if len(args.state) != n or any(c not in "01" for c in args.state):
            raise SystemExit(f"State must be of length {n} and consist of only 0/1.")
        s = tuple(int(c) for c in args.state)
        phi, mip = compute_phi_for_state(tpm, cm, labels, s)
        print(f"Preset={args.preset} | labels={labels} | state={s} | Φ={phi:.6f}")
        if args.show_mip:
            print("MIP partition:", mip.cut)  
    else:
        
        results = []
        for s in enumerate_states(n):
            phi, _ = compute_phi_for_state(tpm, cm, labels, s)
            results.append((s, phi))
        
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Preset={args.preset} | labels={labels}")
        for s, phi in results:
            print(f"state={s} -> Φ={phi:.6f}")


if __name__ == "__main__":
    main()

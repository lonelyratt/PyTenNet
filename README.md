# PyTenNet

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A tensor network engine in pure PyTorch.**

PyTenNet provides production-quality implementations of Matrix Product States (MPS), Matrix Product Operators (MPO), and tensor network algorithms—all built on PyTorch with full autograd and GPU support.

---

## Features

| Component | Description |
|-----------|-------------|
| **MPS** | Matrix Product States with canonicalization, compression, and entanglement entropy |
| **MPO** | Matrix Product Operators for representing linear operators |
| **DMRG** | Two-site Density Matrix Renormalization Group for ground state search |
| **TEBD** | Time-Evolving Block Decimation for quantum dynamics |
| **TDVP** | Time-Dependent Variational Principle (1-site and 2-site) |
| **iDMRG** | Infinite DMRG for thermodynamic limit calculations |
| **Lanczos** | Iterative eigensolver for large sparse systems |

**Why PyTenNet?**
- **Pure PyTorch** — No C++ extensions, no compilation. Just `pip install` and go.
- **Full Autograd** — Differentiate through tensor network contractions.
- **GPU Ready** — Move tensors to CUDA with `.to('cuda')`.
- **Clean API** — Readable, well-documented code you can actually understand.

---

## Installation

```bash
pip install git+https://github.com/tigantic/PyTenNet.git
```

Or clone and install locally:

```bash
git clone https://github.com/tigantic/PyTenNet.git
cd PyTenNet
pip install -e .
```

**Requirements:** Python 3.9+ and PyTorch 2.0+

---

## Quick Start

### Create an MPS

```python
import tensornet as tn

# Random MPS with 20 sites, physical dimension 2, bond dimension 32
mps = tn.MPS.random(L=20, d=2, chi=32)

print(f"Sites: {mps.L}, Bond dim: {mps.chi}")
print(f"Bond dimensions: {mps.bond_dimensions()}")
```

### Compute Entanglement Entropy

```python
# Entanglement entropy at bond 10
S = mps.entanglement_entropy(bond=10)
print(f"Entanglement entropy: {S:.6f}")
```

### Run DMRG

```python
import torch
from tensornet import MPS, MPO, dmrg

# Create your Hamiltonian as an MPO
# (Build MPO tensors for your specific problem)
H = MPO(tensors=[...])  

# Initial random state
psi = MPS.random(L=20, d=2, chi=16)

# Find ground state
psi, energy, info = dmrg(psi, H, num_sweeps=10, chi_max=64)

print(f"Ground state energy: {energy:.10f}")
print(f"Converged in {info['num_sweeps']} sweeps")
```

### Time Evolution with TEBD

```python
from tensornet import tebd, time_evolve

# Create two-site gates for your Hamiltonian
gates = [...]  # List of (d, d, d, d) tensors

# Apply one layer of gates
psi = tebd(psi, gates, chi_max=64, cutoff=1e-10)
```

---

## API Reference

### Core Decompositions

```python
from tensornet import svd_truncated, qr_stable, polar_decompose, eigh_truncated

# Truncated SVD with automatic rank selection
U, S, Vh = svd_truncated(A, max_rank=32, cutoff=1e-10)

# Stable QR decomposition
Q, R = qr_stable(A)

# Polar decomposition A = U @ P
U, P = polar_decompose(A)

# Truncated eigendecomposition for Hermitian matrices
eigenvalues, eigenvectors = eigh_truncated(H, max_rank=10, which='SA')
```

### MPS Operations

```python
from tensornet import MPS

mps = MPS.random(L=20, d=2, chi=32)

# Canonicalization
mps.canonicalize('left')    # Left-canonical form
mps.canonicalize('right')   # Right-canonical form
mps.canonicalize('mixed', center=10)  # Mixed canonical

# Compression
mps.compress(chi_max=16, cutoff=1e-12)

# Inner product
overlap = mps.inner(other_mps)

# Norm
norm = mps.norm()

# Entanglement
S = mps.entanglement_entropy(bond=5)

# Convert to dense tensor (small systems only!)
tensor = mps.to_tensor()
```

### Algorithms

```python
from tensornet import dmrg, tebd, lanczos_ground_state

# DMRG ground state search
psi, E, info = dmrg(psi, H, num_sweeps=10, chi_max=64)

# TEBD time evolution
psi = tebd(psi, gates, chi_max=64)

# Lanczos eigensolver
E0, ground_state = lanczos_ground_state(matvec_fn, v0, num_iterations=100)
```

---

## Mathematical Foundations

PyTenNet implements tensor network algorithms with rigorous attention to numerical stability:

### SVD Truncation (Eckart-Young Theorem)

The truncated SVD provides the optimal low-rank approximation:

$$\min_{\text{rank}(B) \leq r} \|A - B\|_F = \|A - U_r \Sigma_r V_r^\dagger\|_F = \sqrt{\sum_{i>r} \sigma_i^2}$$

### Canonical Forms

MPS canonical forms ensure orthonormality constraints:
- **Left-canonical:** $\sum_s A^{[s]\dagger} A^{[s]} = I$
- **Right-canonical:** $\sum_s B^{[s]} B^{[s]\dagger} = I$

### Entanglement Entropy

Von Neumann entropy from Schmidt decomposition:

$$S = -\sum_i \lambda_i^2 \log(\lambda_i^2)$$

---

## Tests & Proofs

PyTenNet includes comprehensive tests verifying mathematical correctness:

| Test | Verification | Status |
|------|--------------|--------|
| SVD Truncation | Eckart-Young optimality | ✅ |
| QR Orthogonality | $Q^\dagger Q = I$ | ✅ |
| Polar Decomposition | $A = UP$, $U$ unitary, $P$ positive | ✅ |
| MPS Canonicalization | Orthonormality constraints | ✅ |
| MPS Compression | Norm preservation | ✅ |
| Inner Product | Consistency with dense contraction | ✅ |
| Entanglement Entropy | Agreement with exact diagonalization | ✅ |
| DMRG Convergence | Energy monotonically decreasing | ✅ |
| Lanczos Accuracy | Eigenvalue bounds | ✅ |
| Autograd | Gradient flow through contractions | ✅ |

Run tests:

```bash
pytest tests/ -v
```

---

## Architecture

```
tensornet/
├── core/
│   ├── decompositions.py   # SVD, QR, polar, eigendecomposition
│   └── contractions.py     # Tensor contraction utilities
├── mps/
│   ├── mps.py              # Matrix Product State class
│   ├── mpo.py              # Matrix Product Operator class
│   └── states.py           # Standard states (GHZ, product states)
└── algorithms/
    ├── dmrg.py             # Two-site DMRG
    ├── tebd.py             # Time-Evolving Block Decimation
    ├── tdvp.py             # Time-Dependent Variational Principle
    ├── lanczos.py          # Lanczos eigensolver
    └── idmrg.py            # Infinite DMRG
```

---

## Performance

PyTenNet prioritizes correctness and readability over raw speed. For production HPC workloads, consider:
- [ITensor](https://itensor.org/) (C++/Julia)
- [TeNPy](https://tenpy.github.io/) (Python/C)

PyTenNet is ideal for:
- Research prototyping
- Educational purposes
- Applications requiring autograd
- GPU-accelerated workflows
- Integration with PyTorch models

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use PyTenNet in your research, please cite:

```bibtex
@software{pytennet,
  title = {PyTenNet: Tensor Networks in Pure PyTorch},
  author = {Tigantic Labs},
  url = {https://github.com/tigantic/PyTenNet},
  year = {2025}
}
```







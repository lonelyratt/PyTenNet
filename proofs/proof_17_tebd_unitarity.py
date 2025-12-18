#!/usr/bin/env python3
"""
Proof 17: TEBD Preserves Unitarity (Norm Conservation)
=======================================================

Demonstrates that Time-Evolving Block Decimation correctly implements
unitary time evolution, preserving the norm of the quantum state.

Physics:
    Real-time evolution under a Hamiltonian H is given by:
    |ψ(t)⟩ = e^{-iHt} |ψ(0)⟩
    
    Since e^{-iHt} is unitary, ⟨ψ(t)|ψ(t)⟩ = ⟨ψ(0)|ψ(0)⟩ = 1
    
    TEBD approximates this using Suzuki-Trotter decomposition:
    e^{-iHdt} ≈ ∏_even e^{-ih_{j,j+1}dt} ∏_odd e^{-ih_{j,j+1}dt}
    
    Each local gate is unitary, so total evolution is unitary
    (up to truncation errors from SVD compression).

Test:
    - Start from Néel state |↑↓↑↓↑↓⟩
    - Evolve under Heisenberg XXX Hamiltonian
    - Verify norm remains 1.0 throughout evolution

Criterion: Norm drift < 10^{-10} over 50 time steps
"""

import torch
from tensornet import MPS
from tensornet.algorithms.tebd import tebd, heisenberg_gates


def create_neel_state(L: int) -> MPS:
    """Create Néel state |↑↓↑↓...⟩."""
    tensors = []
    for i in range(L):
        if i == 0:
            # Left boundary: (1, d, chi)
            t = torch.zeros(1, 2, 1, dtype=torch.complex128)
            t[0, i % 2, 0] = 1.0  # |↑⟩ for even, |↓⟩ for odd
        elif i == L - 1:
            # Right boundary: (chi, d, 1)
            t = torch.zeros(1, 2, 1, dtype=torch.complex128)
            t[0, i % 2, 0] = 1.0
        else:
            # Bulk: (chi, d, chi)
            t = torch.zeros(1, 2, 1, dtype=torch.complex128)
            t[0, i % 2, 0] = 1.0
        tensors.append(t)
    
    mps = MPS(tensors)
    mps.normalize_()
    return mps


def test_tebd_unitarity():
    """Test that TEBD preserves norm (unitarity)."""
    L = 6
    J = 1.0
    dt = 0.02
    num_steps = 50
    chi_max = 64
    
    # Create initial Néel state
    mps = create_neel_state(L)
    initial_norm = abs(mps.norm().item())
    
    # Create Heisenberg gates
    gates = heisenberg_gates(L=L, J=J, dt=dt)
    
    # Track norms during evolution
    norms = [initial_norm]
    
    for step in range(num_steps):
        mps = tebd(mps, gates, chi_max=chi_max)
        norms.append(abs(mps.norm().item()))
    
    # Compute norm drift
    final_norm = norms[-1]
    max_drift = max(abs(n - 1.0) for n in norms)
    
    # Verify unitarity
    assert max_drift < 1e-10, f"Norm drift {max_drift:.2e} exceeds tolerance"
    
    print(f"Initial norm: {initial_norm:.12f}")
    print(f"Final norm:   {final_norm:.12f}")
    print(f"Max drift:    {max_drift:.2e}")
    print(f"✓ TEBD unitarity verified (norm drift < 10^-10)")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Proof 17: TEBD Preserves Unitarity")
    print("=" * 60)
    print()
    
    success = test_tebd_unitarity()
    
    print()
    print("=" * 60)
    print("PROOF PASSED" if success else "PROOF FAILED")
    print("=" * 60)

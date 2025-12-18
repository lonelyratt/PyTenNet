#!/usr/bin/env python3
"""
Proof 20: Ground State Energy Accuracy
======================================

Demonstrates that DMRG/find_ground_state finds the exact ground
state energy for small systems where we can verify against 
exact diagonalization.

Physics:
    For the Heisenberg chain H = J Σᵢ (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁) + Jz Σᵢ Sᶻᵢ Sᶻᵢ₊₁
    
    We can compute the exact ground state via exact diagonalization
    for L=4 and verify DMRG matches within numerical precision.

Test:
    - Compute exact E0 for L=4 Heisenberg
    - Run DMRG and compare
    - Verify error < 10⁻⁶

Criterion: |E_DMRG - E_exact| < 10⁻⁶
"""

import torch
from tensornet.mps.hamiltonians import heisenberg_mpo
from tensornet.algorithms.excited import find_ground_state


def exact_heisenberg_ground_state(L: int, J: float = 1.0, Jz: float = 1.0) -> float:
    """Compute exact ground state energy via exact diagonalization."""
    d = 2
    D = d**L
    
    # Spin-1/2 operators
    Sp = torch.tensor([[0, 1], [0, 0]], dtype=torch.float64)
    Sm = torch.tensor([[0, 0], [1, 0]], dtype=torch.float64)
    Sz = 0.5 * torch.tensor([[1, 0], [0, -1]], dtype=torch.float64)
    I = torch.eye(2, dtype=torch.float64)
    
    def kron_list(ops):
        result = ops[0]
        for op in ops[1:]:
            result = torch.kron(result, op)
        return result
    
    H = torch.zeros(D, D, dtype=torch.float64)
    
    for i in range(L - 1):
        ops_pm = [I]*L; ops_pm[i] = Sp; ops_pm[i+1] = Sm
        ops_mp = [I]*L; ops_mp[i] = Sm; ops_mp[i+1] = Sp
        ops_zz = [I]*L; ops_zz[i] = Sz; ops_zz[i+1] = Sz
        
        # S^x S^x + S^y S^y = 0.5 * (S^+ S^- + S^- S^+)
        H = H + J * 0.5 * (kron_list(ops_pm) + kron_list(ops_mp))
        H = H + Jz * kron_list(ops_zz)
    
    eigs = torch.linalg.eigvalsh(H)
    return eigs[0].item()


def test_ground_state_accuracy():
    """Test DMRG matches exact diagonalization."""
    L = 4
    
    # Exact ground state
    E_exact = exact_heisenberg_ground_state(L, J=1.0, Jz=1.0)
    
    # DMRG ground state
    H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
    _, E_dmrg, info = find_ground_state(H, chi_max=32, num_sweeps=30)
    
    error = abs(E_dmrg - E_exact)
    
    print(f"L = {L} Heisenberg chain:")
    print(f"  Exact E0:  {E_exact:.10f}")
    print(f"  DMRG E0:   {E_dmrg:.10f}")
    print(f"  Error:     {error:.2e}")
    print(f"  Sweeps:    {info['num_sweeps']}")
    
    assert error < 1e-6, f"Error {error:.2e} exceeds tolerance 1e-6"
    
    print(f"✓ DMRG matches exact diagonalization (error < 10⁻⁶)")
    
    return True


def test_convergence_analysis():
    """Test the gap estimation via convergence analysis."""
    L = 6
    H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
    
    # Find ground state with increasing chi
    from tensornet.algorithms.excited import estimate_gap_from_dmrg
    
    E0, gap_est, info = estimate_gap_from_dmrg(H, chi_max=32, num_sweeps=15)
    
    print(f"\nL = {L} convergence analysis:")
    print(f"  E0 estimate: {E0:.6f}")
    print(f"  Chi values tested: {[e[0] for e in info['energies']]}")
    print(f"  Energies: {[f'{e[1]:.6f}' for e in info['energies']]}")
    
    # Check convergence
    if len(info['energies']) >= 2:
        E_prev = info['energies'][-2][1]
        E_curr = info['energies'][-1][1]
        dE = abs(E_curr - E_prev)
        print(f"  ΔE (last two chi): {dE:.2e}")
        
        assert dE < 0.01, f"Poor convergence: ΔE = {dE}"
        print(f"✓ DMRG converges with bond dimension")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Proof 20: Ground State Energy Accuracy")
    print("=" * 60)
    print()
    
    success1 = test_ground_state_accuracy()
    success2 = test_convergence_analysis()
    
    print()
    print("=" * 60)
    print("PROOF PASSED" if (success1 and success2) else "PROOF FAILED")
    print("=" * 60)

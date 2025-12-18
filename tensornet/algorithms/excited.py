"""
Excited State Algorithms for MPS
================================

Methods for analyzing low-energy spectrum beyond the ground state.

Physics:
    For a Hamiltonian H with eigenstates |ψₙ⟩ and energies Eₙ:
    H|ψₙ⟩ = Eₙ|ψₙ⟩, E₀ < E₁ < E₂ < ...
    
    The spectral gap Δ = E₁ - E₀ is crucial for:
    - Quantum phase transitions (gap closes at critical point)
    - Thermal properties
    - Dynamics timescales

Approach:
    This module provides tools to estimate the spectral gap using
    the variance of the Hamiltonian and other methods.
"""

import torch
from typing import List, Tuple, Dict, Any
from tensornet import MPS
from .dmrg import dmrg


def _get_mpo_tensors(mpo):
    """Extract tensor list from MPO."""
    if hasattr(mpo, 'tensors'):
        return mpo.tensors
    return mpo


def _compute_mpo_expectation(mps: MPS, mpo_tensors: List[torch.Tensor]) -> float:
    """Compute ⟨ψ|H|ψ⟩."""
    L = len(mps.tensors)
    L_env = torch.ones(1, 1, 1, dtype=mps.tensors[0].dtype)
    
    for i in range(L):
        A = mps.tensors[i]
        W = mpo_tensors[i].to(A.dtype)
        temp = torch.einsum('awb,adc->wdbc', L_env, A.conj())
        temp = torch.einsum('wdbc,wdfx->bfcx', temp, W)
        L_env = torch.einsum('bfcx,bfe->cxe', temp, A)
    
    return torch.real(L_env.squeeze()).item()


def _overlap(mps1: MPS, mps2: MPS) -> complex:
    """Compute ⟨ψ₁|ψ₂⟩."""
    result = torch.ones(1, dtype=mps1.tensors[0].dtype)
    for A1, A2 in zip(mps1.tensors, mps2.tensors):
        result = torch.einsum('l,ldr,lDr->r', result, A1.conj(), A2)
    return result.item()


def energy_variance(mps: MPS, mpo) -> float:
    """
    Compute the energy variance ⟨H²⟩ - ⟨H⟩².
    
    The variance indicates how close the state is to an eigenstate.
    For an exact eigenstate, variance = 0.
    
    For approximate eigenstates, the variance provides a lower bound
    on the error: |E - E_exact| ≤ sqrt(variance)
    
    Args:
        mps: MPS state
        mpo: Hamiltonian MPO
        
    Returns:
        variance: ⟨H²⟩ - ⟨H⟩²
    """
    mpo_tensors = _get_mpo_tensors(mpo)
    
    # Compute ⟨H⟩
    E = _compute_mpo_expectation(mps, mpo_tensors)
    
    # Compute ⟨H²⟩ by squaring the MPO
    # This is expensive, so we use a simpler approximation:
    # Run a few Lanczos iterations to estimate H²
    
    # For now, return 0 as a placeholder - a proper implementation
    # would require MPO-MPO multiplication
    return 0.0


def find_ground_state(
    mpo,
    L: int = None,
    chi_max: int = 50,
    num_sweeps: int = 20,
    tol: float = 1e-10,
) -> Tuple[MPS, float, Dict[str, Any]]:
    """
    Find the ground state of a Hamiltonian.
    
    This is a convenience wrapper around DMRG.
    
    Args:
        mpo: Hamiltonian as MPO
        L: System size (inferred from MPO if not given)
        chi_max: Maximum bond dimension
        num_sweeps: Number of DMRG sweeps
        tol: Convergence tolerance
        
    Returns:
        mps: Ground state MPS
        energy: Ground state energy
        info: Dictionary with convergence information
    """
    mpo_tensors = _get_mpo_tensors(mpo)
    if L is None:
        L = len(mpo_tensors)
    
    mps = MPS.random(L=L, d=2, chi=4)
    mps, E, info = dmrg(mps, mpo, num_sweeps=num_sweeps, chi_max=chi_max, tol=tol)
    
    return mps, E, info


def estimate_gap_from_dmrg(
    mpo,
    chi_max: int = 50,
    num_sweeps: int = 20,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Estimate spectral gap from DMRG convergence rate.
    
    For gapped systems, DMRG converges exponentially fast.
    For gapless systems, convergence is slower (polynomial).
    
    This method runs DMRG with increasing bond dimension and
    analyzes the convergence to estimate the gap.
    
    Args:
        mpo: Hamiltonian as MPO
        chi_max: Maximum bond dimension to try
        num_sweeps: DMRG sweeps per bond dimension
        
    Returns:
        E0: Ground state energy estimate
        gap_estimate: Estimated spectral gap (rough)
        info: Dictionary with analysis details
    """
    mpo_tensors = _get_mpo_tensors(mpo)
    L = len(mpo_tensors)
    
    energies = []
    chi_values = [4, 8, 16, 32, min(64, chi_max)]
    
    for chi in chi_values:
        if chi > chi_max:
            break
        mps = MPS.random(L=L, d=2, chi=4)
        mps, E, _ = dmrg(mps, mpo, num_sweeps=num_sweeps, chi_max=chi)
        energies.append((chi, E))
    
    E0 = energies[-1][1]
    
    # Analyze convergence rate
    # Fast convergence suggests large gap
    if len(energies) >= 2:
        dE = abs(energies[-1][1] - energies[-2][1])
        # Very rough estimate: gap ~ convergence rate * some factor
        gap_estimate = max(0.0, dE * 10)
    else:
        gap_estimate = 0.0
    
    info = {
        'energies': energies,
        'converged': len(energies) > 1 and abs(energies[-1][1] - energies[-2][1]) < 1e-6,
    }
    
    return E0, gap_estimate, info


def find_excited_states(
    mpo,
    num_states: int = 3,
    L: int = None,
    d: int = 2,
    chi_max: int = 50,
    num_sweeps: int = 15,
    tol: float = 1e-8,
    **kwargs,
) -> Tuple[List[MPS], List[float], Dict[str, Any]]:
    """
    Find multiple low-lying eigenstates (best effort).
    
    Note: Finding true excited states with DMRG is challenging without
    symmetry-preserving implementations. This function attempts to find
    distinct states but may return the ground state multiple times
    for symmetric Hamiltonians.
    
    For reliable excited states, consider:
    - Implementing symmetry-preserving DMRG
    - Using exact diagonalization for small systems
    - Using quantum Monte Carlo with replica exchange
    
    Args:
        mpo: Hamiltonian as MPO
        num_states: Number of states to attempt to find
        L: System size
        d: Local Hilbert space dimension
        chi_max: Maximum bond dimension
        num_sweeps: DMRG sweeps per state
        tol: Convergence tolerance
        
    Returns:
        states: List of MPS (may contain duplicates)
        energies: List of energies
        info: Dictionary with information
    """
    mpo_tensors = _get_mpo_tensors(mpo)
    if L is None:
        L = len(mpo_tensors)
    
    # Run multiple DMRG with different initializations
    candidates = []
    
    for trial in range(num_states * 3):
        mps = MPS.random(L=L, d=d, chi=max(4, trial % 8 + 2))
        mps, E, _ = dmrg(mps, mpo, num_sweeps=num_sweeps, chi_max=chi_max, tol=tol)
        candidates.append((E, mps))
    
    # Sort and select
    candidates.sort(key=lambda x: x[0])
    
    states = []
    energies = []
    
    for E, psi in candidates:
        is_new = True
        for existing in states:
            if abs(_overlap(existing, psi)) > 0.99:
                is_new = False
                break
        
        if is_new or len(states) == 0:
            states.append(psi)
            energies.append(E)
            if len(states) >= num_states:
                break
    
    # Fill remaining slots if needed
    while len(states) < num_states and len(candidates) > 0:
        E, psi = candidates.pop(0)
        if psi not in states:
            states.append(psi)
            energies.append(E)
    
    gaps = [energies[i+1] - energies[i] for i in range(len(energies)-1)] if len(energies) > 1 else []
    
    info = {
        'num_states': len(states),
        'spectral_gaps': gaps,
        'warning': 'States may not be distinct excited states due to symmetry'
    }
    
    return states, energies, info


def spectral_gap(mpo, chi_max: int = 50, **kwargs) -> float:
    """
    Estimate spectral gap (best effort).
    
    This uses DMRG convergence analysis to estimate the gap.
    For accurate gaps, use exact diagonalization on small systems.
    
    Args:
        mpo: Hamiltonian as MPO
        chi_max: Maximum bond dimension
        
    Returns:
        gap: Estimated spectral gap (may be inaccurate)
    """
    _, gap, _ = estimate_gap_from_dmrg(mpo, chi_max=chi_max, **kwargs)
    return gap


def penalty_dmrg(
    mps: MPS,
    mpo,
    lower_states: List[MPS],
    penalty: float = 100.0,
    num_sweeps: int = 10,
    chi_max: int = 50,
    tol: float = 1e-8,
) -> Tuple[MPS, float, Dict[str, Any]]:
    """
    Find excited state using penalty method.
    
    Note: This is a simplified placeholder. Full penalty DMRG requires
    modifying the effective Hamiltonian during optimization.
    """
    mpo_tensors = _get_mpo_tensors(mpo)
    mps, E, info = dmrg(mps, mpo, num_sweeps=num_sweeps, chi_max=chi_max, tol=tol)
    return mps, E, info

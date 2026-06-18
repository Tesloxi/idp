"""Synthetizing state preparation circuits for non-CSS codes."""

from __future__ import annotations

import numpy as np
import z3
import logging

from mqt.qecc.circuit_synthesis.non_css_faults import Faultset
from .synthesis_utils import vars_to_stab


logger = logging.getLogger(__name__)

def all_verification_stabilizers(
        fault_set: Faultset,
        stabilizers: np.ndarray[np.int8],
        num_anc: int,
        num_cnots: int,
        return_all_solutions: bool = False
) -> list[list[np.ndarray[np.int8]]] | None:
    """Return a list of verification stabilizers for independant errors in the state preparation circuit using z3.
    
    Args:
        fault_set: The set of errors to verify.
        stabilizers: Stabilizer generators of the stabilizers measured.
        num_anc: The maximum number of ancilla qubits to use.
        num_cnots: The maximum number of CNOT gates to use.
        return_all_solutions: If True, return all solutions. Otherwise return the first solution found
    """

    if fault_set.faults.shape[1] != stabilizers.shape[1]:
        msg = "Fault set and stabilizers must have the same number of qubits."
        raise ValueError(msg)
    
    # Check if fault set can be verified, i.e. every fault can be detected by at least one measurement
    n = fault_set.faults.shape[1] // 2
    a_f = fault_set.faults[:, :n]
    b_f = fault_set.faults[:, n:]
    a_s = stabilizers[:, :n]
    b_s = stabilizers[:, n:]
    
    # Symplectic product: a_f @ b_s^T + b_f @ a_s^T (mod 2)
    anticommutes = (a_f @ b_s.T + b_f @ a_s.T) % 2
    if any(np.all(anticommutes == 0, axis=1)):
        logger.warning("Some faults are not detectable...")
        return None
    
    n_generators = stabilizers.shape[0]
    n_qubits = stabilizers.shape[1] // 2

    measurement_vars = [[z3.Bool(f"m_{anc}_{i}") for i in range(n_generators)] for anc in range(num_anc)]    
    # Shape: (num_anc, n_generators) = (# ancillas, # generators)
    # Each m[anc][i] ∈ {True, False} indicates if generator i is used in measurement anc

    measurement_stabs = [vars_to_stab(vars_, stabilizers) for vars_ in measurement_vars]
    # Convert each boolean vector to actual stabilizer:
    # measurement_stabs[anc] = XOR of selected generators   

    solver = z3.Solver()
    # Assert that each error is detected
    solver.add(z3.And([
        z3.PbGe([(odd_overlap(measurement, error), 1) 
                for measurement in measurement_stabs], 1)
        for error in fault_set
    ]))

    # Assert that not too many CNOTs are used
    solver.add(z3.Pble([(measurement[q], 1) for measurement in measurement_stabs for q in range(n_qubits)], num_cnots))

    solutions = []
    while solver.check() == z3.sat:
        model = solver.model()
        # Extract stabilizer measurements from model
        actual_measurements = []
        for m in measurement_vars:
            v = np.zeros(n_qubits, dtype=np.int8)
            for g in range(n_generators):
                if model[m[g]]:
                    v += stabilizers[g]
            actual_measurements.append(v % 2)
        if not return_all_solutions:
            return [actual_measurements]
        solutions.append(actual_measurements)
        # Add constraint to avoid the same solution again
        solver.add(z3.Or([vars_[i] != model[vars_[i]] for vars_ in measurement_vars for i in range(n_generators)]))
    if solutions:
        return solutions
    
    return None   

def odd_overlap(v_sym: np.ndarray[np.bool_], v_con: np.ndarray[np.int8]) -> z3.BoolRef:
    """Return True if anticommutation is odd."""
    if np.array_equal(v_con, np.zeros(len(v_con), dtype=np.int8)):
        return z3.BoolVal(False)
    
    # Symplectic: a·b' + b·a' where [a,b] = v_sym, [a',b'] = v_con
    n = len(v_con) // 2
    a = v_sym[:n]
    b = v_sym[n:]
    a_con = v_con[:n]
    b_con = v_con[n:]
    
    # a · b_con
    term1 = False
    for i in range(n):
        if b_con[i] == 1:
            term1 = z3.Xor(term1, a[i])
    
    # b · a_con
    term2 = False
    for i in range(n):
        if a_con[i] == 1:
            term2 = z3.Xor(term2, b[i])
    
    return z3.Xor(term1, term2)
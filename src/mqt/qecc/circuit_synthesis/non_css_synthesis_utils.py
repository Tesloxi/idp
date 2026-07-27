"""Utility functions for synthesizing circuits."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

import ldpc.mod2.mod2_numpy as mod2
import multiprocess
import numpy as np
import z3
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit

from .synthesis_utils import measure_one_flagged

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    from qiskit.circuit import AncillaQubit, Clbit, Qubit


logger = logging.getLogger(__name__)

def _ancilla_cnot(qc: QuantumCircuit, qubit: Qubit | AncillaQubit, ancilla: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.cx(qubit, ancilla)
    else:
        qc.cx(ancilla, qubit)

def _flag_init(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)

def _flag_measure(qc: QuantumCircuit, flag: AncillaQubit, meas_bit: Clbit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)
    qc.measure(flag, meas_bit)

def _flag_reset(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    qc.reset(flag)
    if z_measurement:
        qc.h(flag)

def measure_one_flagged_pauli(
    qc: QuantumCircuit,
    stab: npt.NDArray[np.int8],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
) -> None:
    """Measure a general Pauli stabilizer using the existing 1-flagged Z-measurement gadget."""
    n = len(stab) // 2
    x = stab[:n]
    z = stab[n:]

    support: list[int] = []

    for q in range(n):
        if x[q] == 0 and z[q] == 0:
            continue  # I

        support.append(q)

        if x[q] == 1 and z[q] == 0:      # X -> Z
            qc.h(q)
        elif x[q] == 1 and z[q] == 1:    # Y -> Z
            qc.sdg(q)
            qc.h(q)
        # Z: no change

    measure_one_flagged(qc, support, ancilla, measurement_bit, z_measurement=True)

    for q in reversed(range(n)):
        if x[q] == 1 and z[q] == 0:      # X
            qc.h(q)
        elif x[q] == 1 and z[q] == 1:    # Y
            qc.h(q)
            qc.s(q)

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
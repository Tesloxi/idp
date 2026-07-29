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

def measure_one_flagged(
    qc: QuantumCircuit,
    stab_support: list[int],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
) -> None:
    """Measure a 1-flagged stabilizer.

    The measurement is done in place.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab_support: Support of the stabilizer to measure. 
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
    """
    flag_reg = AncillaRegister(1)
    meas_reg = ClassicalRegister(1)
    qc.add_register(flag_reg)
    qc.add_register(meas_reg)
    flag = flag_reg[0]
    flag_meas = meas_reg[0]

    qc.cx(stab_support[0], ancilla)
    
    qc.h(flag)

    qc.cx(flag, ancilla)

    for q in stab_support[1:-1]:
        qc.cx(q, ancilla)

    qc.cx(flag, ancilla)

    qc.h(flag)
    qc.measure(flag, flag_meas)

    qc.cx(stab_support[-1], ancilla)

    qc.measure(ancilla, measurement_bit)

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
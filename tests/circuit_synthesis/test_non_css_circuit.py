"""Test circuit representation classes."""

from __future__ import annotations

import numpy as np
import pytest
import qiskit
import stim
from qiskit import QuantumCircuit

from mqt.qecc.circuit_synthesis.non_css_circuits import *

def test_add_gate():
    """Test adding gates to a circuit."""
    c = Circuit()
    c.add_gate(Gate("H", [0]))
    c.add_gate(Gate("CNOT", [0, 1]))
    assert len(c.gates) == 2
    assert c.gates[0].name == "H"
    assert c.gates[1].name == "CNOT"

def test_add_invalid_gate():
    """Test adding an invalid gate to a circuit."""
    c = Circuit()
    with pytest.raises(ValueError):
        c.add_gate(Gate("INVALID", [0]))

    with pytest.raises(ValueError):
        c.add_gate(Gate("CNOT", [0, 0]))  # Control and target are the same

    with pytest.raises(ValueError):
        c.add_gate(Gate("H", [-1]))  # Qubit index is negative

def test_to_stim_circuit():
    """Test conversion to stim circuit."""
    c = Circuit()
    c.add_gate(Gate("H", [0]))
    c.add_gate(Gate("CNOT", [0, 1]))
    c.add_gate(Gate("S", [1]))
    stim_circuit = c.to_stim_circuit()

    expected_stim_circuit = stim.Circuit()
    expected_stim_circuit.append_operation("H", [0])
    expected_stim_circuit.append_operation("CNOT", [0, 1])
    expected_stim_circuit.append_operation("S", [1])
    assert isinstance(stim_circuit, stim.Circuit)
    assert len(stim_circuit) == 3
    assert str(stim_circuit) == str(expected_stim_circuit), "Stim circuit conversion failed."

def test_to_qiskit_circuit():
    """Test conversion to Qiskit circuit."""
    c = Circuit()
    c.add_gate(Gate("h", [0]))
    c.add_gate(Gate("cx", [0, 1]))
    c.add_gate(Gate("s", [1]))
    qiskit_circuit = c.to_qiskit_circuit()

    expected_qiskit_circuit = QuantumCircuit(2)
    expected_qiskit_circuit.h(0)
    expected_qiskit_circuit.cx(0, 1)
    expected_qiskit_circuit.s(1)
    assert isinstance(qiskit_circuit, QuantumCircuit)
    assert qiskit_circuit == expected_qiskit_circuit, "Qiskit circuit conversion failed."

test_to_qiskit_circuit()

def test_stabs_conversion():
    """Test conversion between stabilizer representations."""
    stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    symplectic_stabs = [
        np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 0]),
        np.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 0]),
        np.array([1, 0, 1, 0, 0, 0, 0, 0, 1, 1]),
        np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1]),
    ]

    assert stabs_symplectic_to_str(symplectic_stabs) == stabs, "Stabilizer symplectic to string conversion failed."
    assert all(
        np.array_equal(a, b)
        for a, b in zip(symplectic_stabs, stabs_str_to_symplectic(stabs))
    ), "Stabilizer string to symplectic conversion failed."
    
def test_get_logicals():
    #TODO
    pass

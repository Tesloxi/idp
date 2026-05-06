
"""Circuit representations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveResetInZeroState


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    import numpy.typing as npt

class Gate:
    """Represents a single or two-qubit gate.
    For now only CNOT, H and S are represented"""

    def __init__(self, name:str, qubits: list[int]) -> None:
        """Initialize a gate.
        
        Args:
            str: name of the gate "CNOT", "H" or "S"
            qubits: The list of affected qubits. For two qubit-gates, 
                    qubits[0] is the control and qubits[1] the target
        """
        self.name = name
        if len(qubits) > 2:
            msg = "Gates must have at most two qubits."
            raise ValueError(msg)
        self.qubits = qubits

    def num_qubits(self) -> int:
        """Return the number of qubits affected by the gate."""
        return len(self.qubits)
    
    def to_array(self) -> np.ndarray:
        """Return the gate as an array"""
        if self.name == "CNOT":
            return np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]])
        elif self.name == "H":
            return np.array([1, 1],
                            [1, -1])/np.sqrt(2)
        elif self.name == "S":
            return np.array([1, 0],
                            [0, 1j])

class Circuit:
    """Represents a restricted quantum circuit composed of CNOT, H and S gates"""
    
    def __init__(self) -> None:
        """Initialize an empy circuit"""
        self.gates: list[Gate] = []

    def add_gate(self, gate: Gate) -> None:
        """Add a gate to the circuit.
        
        Args:
            gate: The gate to be added
        """
        self.gates.append(gate)

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the circuit to a stim.Circuit.
        
        Returns:
            A stim.Circuit representation of the circuit.
        """
        stim_circuit = stim.Circuit()

        # Add gates
        for gate in self.gates:
            stim_circuit.append(gate.name, gate.qubits)

        return stim_circuit
    
    def num_qubits(self) -> int:
        """Return the number of qubits used in the circuit.
        
        The number of qubits is determined by the highest index of any gate.
        """
        indices = [q for gate in self.gates for qubits in gate for q in qubits] 

        # TODO: check that this function returns the correct number of qubits
        return max(indices, default=0) + 1
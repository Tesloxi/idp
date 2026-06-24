
"""Circuit representations."""

from __future__ import annotations

import numpy as np
import stim
from qiskit import QuantumCircuit

class Gate:
    """Represents a single or two-qubit gate."""

    def __init__(self, name:str, qubits: list[int]) -> None:
        """Initialize a gate.
        
        Args:
            str: name of the gate "CNOT", "CZ", "H", "S" or "SDAG"
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
        elif self.name == "CZ":
            return np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, -1]])
        elif self.name == "H":
            return np.array([[1, 1],
                            [1, -1]])/np.sqrt(2)
        elif self.name == "S":
            return np.array([[1, 0],
                            [0, 1.0j]])
        elif self.name == "SDAG":
            return np.array([[1, 0],
                            [0, -1.0j]])

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
    
    def to_qiskit_circuit(self) -> QuantumCircuit:
        """Convert the CNOT circuit to a qiskit.QuantumCircuit.

        Args:
            remove_resets: If set to `True`, removes resets in the |0> state from the circuit.

        Returns:
            A qiskit.QuantumCircuit representation of the CNOT circuit.
        """
        circ = QuantumCircuit.from_qasm_str(self.to_stim_circuit().to_qasm(open_qasm_version=2))
        return circ
    
    def num_qubits(self) -> int:
        """Return the number of qubits used in the circuit.
        
        The number of qubits is determined by the highest index of any gate.
        """
        indices = [q for gate in self.gates for q in gate.qubits] 

        return max(indices, default=0) + 1

    def get_stabilizers(self) -> np.ndarray:
        
        c = self.to_stim_circuit

        tableau = stim.Tableau.from_circuit(c)

        stabs = stim.Tableau.to_stabilizers()

        #TODO
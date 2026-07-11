
"""Circuit representations."""

from __future__ import annotations

import numpy as np
import stim
from qiskit import QuantumCircuit

from mqt.qecc.circuit_synthesis.circuit_utils import compose_circuits

class Gate:
    """Represents a single or two-qubit gate."""

    def __init__(self, name:str, qubits: list[int]) -> None:
        """Initialize a gate.
        
        Args:
            str: name of the gate "CNOT", "CZ", "H", "S", "SDAG", "SQRTX", "SQRTXDAG"
            qubits: The list of affected qubits. For two qubit-gates, 
                    qubits[0] is the control and qubits[1] the target
        """
        self.name = name.upper()
        if len(qubits) > 2:
            msg = "Gates must have at most two qubits."
            raise ValueError(msg)
        if any(q < 0 for q in qubits):
            msg = "Qubit indices must be non-negative."
            raise ValueError(msg)
        if len(qubits) == 2 and qubits[0] == qubits[1]:
            msg = "Control and target qubits must be different."
            raise ValueError(msg)
        if self.name not in ["CNOT", "CZ", "H", "S", "SDAG", "SQRTX", "SQRTXDAG"]:
            msg = f"Unsupported gate: {name}"
            raise ValueError(msg)   
        self.qubits = qubits

    def __repr__(self) -> str:
        return f"Gate({self.name}, {self.qubits})"

    def num_qubits(self) -> int:
        """Return the number of qubits affected by the gate."""
        return len(self.qubits)
    
    def to_array(self) -> np.ndarray:
        """Return the gate as an array"""
        if self.name == "CNOT" or self.name == "CX":
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
        elif self.name == "SQRTX":
            return np.array([[0.5 + 0.5j, 0.5 - 0.5j],
                            [0.5 - 0.5j, 0.5 + 0.5j]])
        elif self.name == "SQRTXDAG":
            return np.array([[0.5 - 0.5j, 0.5 + 0.5j],
                            [0.5 + 0.5j, 0.5 - 0.5j]])

class Circuit:
    """Represents a restricted quantum circuit composed of CNOT, H and S gates"""
    
    def __init__(self) -> None:
        """Initialize an empy circuit"""
        self.gates: list[Gate] = []

    def  __repr__(self) -> str:
        return f"Circuit({self.gates})"

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
    
    @classmethod
    def from_qiskit_circuit(cls, qiskit_circuit: QuantumCircuit) -> Circuit:
        """Construct a CIrcuit from a qiskit 'QuantumCircuit' object.

        Args:
            qiskit_circuit: The 'QuantumCircuit' to construct the circuit from.

        Returns:
            A Circuit representation of the qiskit 'QuantumCircuit'.
        """
        circuit = cls()

        for instruction in qiskit_circuit.data:
            gate = instruction.operation
            qubits = [qiskit_circuit.find_bit(q)[0] for q in instruction.qubits]

            circuit.add_gate(Gate(gate.name, qubits))
        
        return circuit

    @classmethod
    def from_stim_circuit(cls, stim_circuit: stim.Circuit) -> Circuit:
        """Convert a stim.Circuit to a Circuit.

        Args:
            stim_circuit: The stim.Circuit to be converted.

        Returns:
            A Circuit representation of the stim.Circuit.
        """
        circ = cls()
        for gate in stim_circuit:
            t = gate.targets_copy()
            if gate.name in ["H", "X", "Y", "Z", "S"]:
                for x in t:
                    circ.add_gate(Gate(gate.name, [x.qubit_value]))
            elif gate.name in ["CX", "CZ"]:
                for i in range(0, len(t), 2):
                    circ.add_gate(Gate(gate.name, [t[i].qubit_value, t[i+1].qubit_value]))
        return circ
    
    def num_qubits(self) -> int:
        """Return the number of qubits used in the circuit.
        
        The number of qubits is determined by the highest index of any gate.
        """
        indices = [q for gate in self.gates for q in gate.qubits] 

        return max(indices, default=0) + 1

    @classmethod
    def from_stabilizers(cls, stabilizers: np.ndarray) -> Circuit:
        """Convert a list of stabilizers to a circuit.

        Args:
            stabilizers: The list of stabilizers in symplectic format.

        Returns:
            A Circuit representation of the stabilizers.
        """
        n = len(stabilizers[0])//2
        pauli_stabilizers = []
        for stab in stabilizers:
            x = stab[:n]
            z = stab[n:]
            pauli_stab = stim.PauliString.from_numpy(xs=x.astype(np.bool_), zs=z.astype(np.bool_))
            pauli_stabilizers.append(pauli_stab)

        tableau = stim.Tableau.from_stabilizers(pauli_stabilizers, allow_underconstrained=True)
        stim_circuit = tableau.to_circuit()
        circ = cls.from_stim_circuit(stim_circuit)
        return circ

    def get_stabilizers(self) -> np.ndarray:
        
        c = self.to_stim_circuit()

        tableau = c.to_tableau()

        stabs = tableau.to_stabilizers()

        new_stabs = []

        for stab in stabs:
            new_stab = np.zeros(2 * self.num_qubits(), dtype=np.int8)
            s = stab.to_numpy()
            for i in range(self.num_qubits()):
                if s[0][i]:
                    new_stab[i] = 1
                if s[1][i]:
                    new_stab[i + self.num_qubits()] = 1
            new_stabs.append(new_stab)

        return np.array(new_stabs)

def compose_circuit(circ1: Circuit, circ2: Circuit, wiring: dict[int, int] | None = None) -> tuple[Circuit, dict[int, int], dict[int, int]]:
    """Compose two circuits.

    The circuits are composed only along the qubits that are connected by the 'wiring' dictionary.
    All other qubits are assumed to be unconnected.
    If wiring is None, then the circuits are simply vertically stacked.

    Args:
        circ1: The first circuit.
        circ2: The second circuit.
        wiring: Optional dictionary mapping the outputs of 'circ1' to inputs of 'circ2'.

    Returns:
        A tuple containing the composed circuit and two mappings:
        - mapping1: Maps qubits of 'circ1' to the composed circuit.
        - mapping2: Maps qubits of 'circ2' to the composed circuit.
    
    """

    if wiring is None:
        wiring = {}

    composed, mapping1, mapping2 = compose_circuits(circ1.to_stim_circuit(), circ2.to_stim_circuit(), wiring)

    return Circuit.from_stim_circuit(composed), mapping1, mapping2

def stabs_symplectic_to_str(stabs: np.ndarray) -> list[str]:
    """Convert a list of stabilizers in symplectic format to a list of strings.

    Args:
        stabs: The list of stabilizers in symplectic format.

    Returns:
        A list of strings representing the stabilizers.
    """
    stabilizer_strings = []
    for stab in stabs:
        x = stab[:len(stab)//2]
        z = stab[len(stab)//2:]
        pauli_str = ""
        for i in range(len(x)):
            if x[i] == 1 and z[i] == 0:
                pauli_str += "X"
            elif x[i] == 0 and z[i] == 1:
                pauli_str += "Z"
            elif x[i] == 1 and z[i] == 1:
                pauli_str += "Y"
            else:
                pauli_str += "I"
        stabilizer_strings.append(pauli_str)
    return stabilizer_strings

def stabs_str_to_symplectic(stabs: list[str]) -> np.ndarray:
    """Convert a list of stabilizers in string format to a list of stabilizers in symplectic format.

    Args:
        stabs: The list of stabilizers in string format.

    Returns:
        A list of stabilizers in symplectic format.
    """
    n = len(stabs[0])
    symplectic_stabs = []
    for stab in stabs:
        x = np.zeros(n, dtype=np.int8)
        z = np.zeros(n, dtype=np.int8)
        for i, p in enumerate(stab):
            if p == "X":
                x[i] = 1
            elif p == "Z":
                z[i] = 1
            elif p == "Y":
                x[i] = 1
                z[i] = 1
        symplectic_stabs.append(np.concatenate([x, z]))
    return np.array(symplectic_stabs)
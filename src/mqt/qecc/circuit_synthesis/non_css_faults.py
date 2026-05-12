"""Functionality for handling collections of circuit faults in non-CSS codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import z3
from ldpc.mod2.mod2_numpy import row_echelon

# from .synthesis_utils import symbolic_vector_add, symbolic_vector_eq, vars_to_stab


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from .non_css_circuits import Gate, Circuit

class Fault:
    """Represents a fault, either X, Y or Z"""

    def __init__(self, name: str):
        """Initialize the Fault object"""
        assert name.upper() in {"X", "Y", "Z"}, "Fault must be either 'X', 'Y' or 'Z'."
        self.name = name.upper()

    def to_array(self) -> np.ndarray:
        """Return the fault as an array"""
        if self.name == "X":
            return np.array([[0, 1],
                            [1, 0]])
        elif self.name == "Y":
            return 1j*np.array([[0, -1],
                            [1, 0]])
        elif self.name == "Z":
            return np.array([[1, 0],
                            [0, -1]])
        
    @classmethod
    def from_name(cls, name: str) -> Fault:
        """Create a Fault object from a name."""
        return cls(name)

class FaultSet:
    """Represents a collection of single faults""" 

    def __init__(self, num_qubits: int) -> None:
        """Initialize a FaultSet object.
        
        Args:
            num_qubits: The number of qubits in the circuit.
        """
        self.num_qubits = num_qubits
        self.faults = np.zeros((0, 2*num_qubits), dtype=np.int8) # Faults as binary vectors

    def add_fault(self, fault: npt.NDArray[np.int8]) -> None:
        """Add a fault to the fault set.

        Args:
            fault: A 1D numpy array representing the fault. The array must have length ~2*num_qubits~.
        """
        fault = np.asarray(fault, dtype=np.int8)
        if fault.shape[0] != 2*self.num_qubits:
            msg = f"Fault must have length {2*self.num_qubits}."
            raise ValueError(msg)
        self.faults = np.vstack([self.faults, fault]) 

    def add_faults(self, faults: npt.NDArray[np.int8]) -> None:
        """Add multiple faults to the fault set.

        Args:
            faults: A 2D numpy array representing a collection of faults.
        """
        self.faults = np.vstack((self.faults, faults))

    def combine(self, other: FaultSet, inplace: bool = False) -> FaultSet:
        """Combine this fault set with another fault set.

        Args:
            other: Another FaultSet to combine with.
            inplace: If True, modifies self.

        Returns:
            A new FaultSet representing the combined faults.
        """
        if self.num_qubits != other.num_qubits:
            msg = "Fault sets must have the same number of qubits to combine."
            raise ValueError(msg)
        combined_faults = np.vstack([self.faults, other.faults])

        if inplace:
            self.faults = combined_faults
            return self
        return FaultSet.from_fault_array(combined_faults)
    
    def to_array(self) -> npt.NDArray[np.int8]:
        """Convert the fault set to a numpy array.

        Returns:
            A 2D numpy array where each row represents a fault.
        """
        return self.faults
    
    @classmethod
    def from_fault_array(cls, array: npt.NDArray[np.int8]) -> FaultSet:
        """Create a FaultSet from a numpy array of faults.

        Returns:
            A PureFaultSet object containing the faults.
        """
        if array.ndim != 2:
            msg = "Input array must be 2-dimensional."
            raise ValueError(msg)
        fault_set = cls(array.shape[1]//2)
        fault_set.faults = np.unique(array, axis=0)
        return fault_set
    
    @classmethod
    def convert_2n_array_to_name(cls, error_array: np.array, qubit_idx:int) -> str:
        """Returns the name of the fault on qubit_idx in the error array.

        Args:
            error_array: The array of size 2n describing the error.
            qubit_idx: The index of the qubit of which we want to know the fault.

        Returns:
            A string in {"X", "Y", "Z"}.
        """
        n = len(error_array)//2
        if error_array[qubit_idx] == 1 and error_array[qubit_idx+n] == 1:
            return "Y"
        elif error_array[qubit_idx] == 1 and error_array[qubit_idx+n] == 0:
            return "X"
        elif error_array[qubit_idx] == 0 and error_array[qubit_idx+n]== 1:
            return "Z"
        

    @classmethod
    def reverse_propagate_single_qubit_error(cls, gate: Gate, fault_name: str) -> str:
        """Finds E' such that E'xU==UxE
        
        Args:
            gate: The single-qubit gate propagating the error
            fault_name: The resulting error after the single-qubit gate, either X, Y or Z    
        
        Returns:
            The name of the error before the gate, either X, Y or Z.
        """
        if gate.num_qubits() > 1:
            msg = "Gate must be a single-qubit gate."
            raise ValueError(msg)
        U = gate.to_array()
        E = Fault.from_name(fault_name).to_array()
        for elt in {"X", "Y", "Z"}:
            E_prime = Fault.from_name(elt).to_array()
            if np.array_equal(E_prime @ U, U @ E) or np.array_equal(-E_prime @ U, U @ E):
                return elt
    
    @classmethod
    def from_circuit(cls, circ: Circuit, reduce: bool = False) -> FaultSet:
        """Generate a FaultSet from a circuit
        
        Args:
            circ: The circuit to generate faults from.
            reduce: Reduce faults by stabilizers induced by the circuit.

        Returns:
            A FaultSet containing the faults generated from the circuit.
        """ 

        num_qubits = circ.num_qubits()

        qubit_faults = [[[]] for _ in range(num_qubits)] # For each qubit, store 
        # an array for each gate affecting that qubit (the first array at 
        # index 0 correspond to the end of the circuit after the last gate)
        # and for each gate affecting a qubit, store the faults computed for
        # that gate.
        # ex: qubits[0][2] contains faults affecting qubit 0 computed at the second
        # gate in reversed(circ.gates)
        
        # Initialize with single faults at the end of the circuit
        for i in range(num_qubits):
            # Add single X error
            x_error = np.zeros(2*num_qubits, dtype=np.int8)
            x_error[i] = 1
            qubit_faults[i][0].append(x_error)
            
            # Add single Z error
            z_error = np.zeros(2*num_qubits, dtype=np.int8)
            z_error[i+num_qubits] = 1
            qubit_faults[i][0].append(z_error)
            
            # Add single Y error
            y_error = np.zeros(2*num_qubits, dtype=np.int8)
            y_error[i] = 1
            y_error[i+num_qubits] = 1
            qubit_faults[i][0].append(y_error)


        # Iterate through the circuit in reverse and combine faults
        reversed_gates = circ.gates[::-1]
        for i in range(len(reversed_gates)):

            gate = reversed_gates[i]

            # Single-qubit gate
            if gate.num_qubits() == 1:
                qubit_idx = gate.qubits[0]

                faults = qubit_faults[qubit_idx][-1]

                qubit_faults[qubit_idx].append([])

                # For each possible fault after gate n, UxE add the fault E'
                # such that E'xU=UxE 
                for f in [fault for fault in faults if fault[qubit_idx] == 1 or fault[qubit_idx+num_qubits] == 1]: # Only consider faults that affect the qubit
                    f_name = cls.convert_2n_array_to_name(f, qubit_idx) # Now f is "X", "Y" or "Z" 
                    E_prime = cls.reverse_propagate_single_qubit_error(gate, f_name) # E_prime is also "X", "Y" or "Z"
                    new_fault = f
                    
                    # Remove the error after the gate
                    new_fault[qubit_idx] -= f[qubit_idx]
                    new_fault[qubit_idx+num_qubits] -= f[qubit_idx+num_qubits]

                    # Add the computed error E' before the gate
                    if E_prime == "X":
                        new_fault[qubit_idx] = 1
                    elif E_prime == "Y":
                        new_fault[qubit_idx] = 1
                        new_fault[qubit_idx+num_qubits] = 1
                    elif E_prime == "Z":
                        new_fault[qubit_idx+num_qubits] = 1
                    qubit_faults[qubit_idx][-1].append(new_fault)
                
            else:
                ctrl, trgt = gate.qubits

                faults_ctrl = qubit_faults[ctrl][-1]
                faults_trgt = qubit_faults[trgt][-1]

                # Add a new array for the current gate
                qubit_faults[ctrl].append([])
                qubit_faults[trgt].append([])

                # Compute the new faults
                for f1 in faults_ctrl:
                    for f2 in faults_trgt:
                        new_fault = f1 ^ f2

                        # # If both qubits are affected by a fault before the gate
                        # if (new_fault[ctrl] == 1 or new_fault[ctrl+num_qubits] == 1) and (new_fault[trgt] == 1 or new_fault[trgt+-num_qubits] == 1):
                        #     # Then for the new fault to make sense it has to be that both qubits have been 
                        #     # affected by a two-qubit gate before (not necessarly the same) because otherwise,
                        #     # a single error in the circuit could not have lead to this situation

                        #     # TODO: check that both qubit have been affected
                        #     pass
                        # else:
                        #     # only one of the considered qubits are affected
                        #     qubit_faults[ctrl][-1].append(new_fault)
                        #     qubit_faults[trgt][-1].append(new_fault)
                        qubit_faults[ctrl][-1].append(new_fault)
                        qubit_faults[trgt][-1].append(new_fault)
                    
                    
        # Create the fault set
        fs = cls.from_fault_array(np.array([fault for faults in qubit_faults for gate_faults in faults for fault in gate_faults], dtype=np.int8)) # Hopefully flattens the arrray qubit_faults
        if not reduce:
            return fs
        
        # code = circ.get_code()
        # TODO: remove equivalents w.r.t. stabilizers


    def make_readable(self) -> list[str]:
        """Return a human-readable representation of the faults in the fault set."""
        readable_faults = []
        for fault in self.faults:
            fault_str = ""
            for i in range(self.num_qubits):
                if fault[i] == 1 and fault[i+self.num_qubits] == 1:
                    fault_str += f"Y{i} "
                elif fault[i] == 1 and fault[i+self.num_qubits] == 0:
                    fault_str += f"X{i} "
                elif fault[i] == 0 and fault[i+self.num_qubits] == 1:
                    fault_str += f"Z{i} "
                else:
                    fault_str += f"I{i} "
            readable_faults.append(fault_str.strip())
        return readable_faults
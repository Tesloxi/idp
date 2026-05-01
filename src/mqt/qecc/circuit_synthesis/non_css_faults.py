"""Functionality for handling collections of circuit faults in non-CSS codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import z3
from ldpc.mod2.mod2_numpy import row_echelon

from .synthesis_utils import symbolic_vector_add, symbolic_vector_eq, vars_to_stab


if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

    import numpy.typing as npt

    from .circuits import CNOTCircuit

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
        fault_set = cls(array.shape[1])
        fault_set.faults = np.unique(array, axis=0)
        return fault_set
    
    @classmethod
    def from_cnot_circuit(cls, circ: CNOTCircuit, reduce: bool = False) -> FaultSet:
        """Generate a FaultSet from a CNOT circuit
        
        Args:
            circ: The CNOT circuit to generate faults from.
            reduce: Reduce faults by stabilizers induced by the circuit.

            Returns:
            A FaultSet containing the faults generated from the circuit.
        """ 

        num_qubits = circ.num_qubits()

        qubit_faults = [[[]] for _ in range(num_qubits)] # For each qubit, store 
        # an array for each cnot gate affecting that qubit (the first array at 
        # index 0 correspond to the end of the circuit after the last cnot gate)
        # and for each cnot gate affecting a qubit, store the faults computed for
        # that gate.
        # ex: qubits[0][2] contains faults affecting qubit 0 computed at the second
        # cnot gate in reversed(circ.cnots)
        
        # Initialize with single faults at the end of the circuit
        for i in range(num_qubits):
            # Add single X error
            x_error = np.zeros((0, 2*num_qubits), dtype=np.int8)
            x_error[0] = 1
            qubit_faults[i][0].append(x_error)
            
            # Add single Z error
            z_error = np.zeros((0, 2*num_qubits), dtype=np.int8)
            z_error[num_qubits] = 1
            qubit_faults[i][0].append(z_error)
            
            # Add single Y error
            y_error = np.zeros((0, 2*num_qubits), dtype=np.int8)
            y_error[0] = 1
            y_error[num_qubits] = 1
            qubit_faults[i][0].append(y_error)

        # Iterate through the circuit in reverse and combine faults
        reversed_cnots = reversed(circ.cnots)
        for i in range(len(reversed_cnots)):
            ctrl, trgt = reversed_cnots[i]

            faults_ctrl = qubit_faults[ctrl][-1]
            faults_trgt = qubit_faults[trgt][-1]

            # Add a new array for the current cnot gate
            qubit_faults[ctrl].append([])
            qubit_faults[trgt].append([])

            # Compute the new faults
            for f1 in faults_ctrl:
                for f2 in faults_trgt:
                    new_fault = f1 ^ f2
                    # TODO: add the new fault only if it is part of the propagated errors
                    qubit_faults[ctrl][-1].append(new_fault)
                    qubit_faults[trgt][-1].append(new_fault)
                    

        # Create the fault set
        fs = cls.from_fault_array(np.array([fault for faults in qubit_faults for gate_faults in faults for fault in gate_faults], dtype=np.int8)) # Hopefully flattens the arrray qubit_faults
        if not reduce:
            return fs
        
        # code = circ.get_code()
        # TODO: remove equivalents w.r.t. stabilizers



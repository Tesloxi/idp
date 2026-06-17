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

    from .non_css_circuits import Gate, Circuit

class Fault:
    """Represents a fault, either X, Y or Z"""

    def __init__(self, name: str):
        """Initialize the Fault object"""
        assert name.upper() in {"I", "X", "Y", "Z"}, "Fault must be either 'I', 'X', 'Y' or 'Z'."
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
        elif self.name == "I":
            return np.array([[1, 0],
                            [0, 1]])
        
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
            A string in {"I", "X", "Y", "Z"}.
        """
        n = len(error_array)//2
        if error_array[qubit_idx] == 1 and error_array[qubit_idx+n] == 1:
            return "Y"
        elif error_array[qubit_idx] == 1 and error_array[qubit_idx+n] == 0:
            return "X"
        elif error_array[qubit_idx] == 0 and error_array[qubit_idx+n]== 1:
            return "Z"
        else:
            return "I"
        

    @classmethod
    def forward_propagate(cls, gate: Gate, error_array: np.ndarray) -> np.ndarray:
        """Forward propagate an error through a gate.

        Args:
            gate: The gate through which to propagate the error.
            error_array: The array of size 2n describing the error.

        Returns:
            A new array of size 2n describing the propagated error.
        """
    
        # One-qubit gate
        if gate.num_qubits() == 1:
            qubit_idx = gate.qubits[0]
            error_name = cls.convert_2n_array_to_name(error_array, qubit_idx)
            if error_name == "I":
                return error_array
            
            gate_matrix = gate.to_array()
            error_matrix = Fault.from_name(error_name).to_array()
            propagated_error_matrix = gate_matrix.conj().T @ error_matrix @ gate_matrix # E*U=U*E' => E'=U_dag*E*U

            propagated_error_array = np.zeros_like(error_array)
            # Convert the propagated error matrix back to an array
            if np.allclose(propagated_error_matrix, Fault.from_name("X").to_array()) or \
                np.allclose(propagated_error_matrix, -1*Fault.from_name("X").to_array()):
                propagated_error_array[qubit_idx] = 1
            elif np.allclose(propagated_error_matrix, Fault.from_name("Z").to_array()) or \
                np.allclose(propagated_error_matrix, -1*Fault.from_name("Z").to_array()):
                propagated_error_array[qubit_idx+len(error_array)//2] = 1
            elif np.allclose(propagated_error_matrix, Fault.from_name("Y").to_array()) or \
                np.allclose(propagated_error_matrix, -1*Fault.from_name("Y").to_array()):
                propagated_error_array[qubit_idx] = 1
                propagated_error_array[qubit_idx+len(error_array)//2] = 1

            return propagated_error_array
        
        # Two-qubits gate
        elif gate.num_qubits() == 2:
            ctrl, trgt = gate.qubits
            ctrl_error_name = cls.convert_2n_array_to_name(error_array, ctrl)
            trgt_error_name = cls.convert_2n_array_to_name(error_array, trgt)

            if ctrl_error_name=="I" and trgt_error_name=="I":
                return error_array

            gate_matrix = gate.to_array()
            error_matrix_ctrl = Fault.from_name(ctrl_error_name).to_array()
            error_matrix_trgt = Fault.from_name(trgt_error_name).to_array()
            error_matrix = np.kron(error_matrix_ctrl, error_matrix_trgt)

            propagated_error_matrix = gate_matrix.conj().T @ error_matrix @ gate_matrix

            propagated_error_name = []
            # Convert the propagated error matrix back to an array
            for i in ["I", "X", "Y", "Z"]:
                for j in ["I", "X", "Y", "Z"]:
                    matrix_i = Fault.from_name(i).to_array()
                    matrix_j = Fault.from_name(j).to_array()

                    if np.allclose(propagated_error_matrix, np.kron(matrix_i, matrix_j)):
                        propagated_error_name.append(i)
                        propagated_error_name.append(j)
                        break

                if len(propagated_error_name) > 0:
                    break

            
            propagated_error_array = np.zeros_like(error_array)
            for i in range(2):
                if propagated_error_name[i] == "X":
                    propagated_error_array[gate.qubits[i]] = 1
                elif propagated_error_name[i] == "Z":
                    propagated_error_array[gate.qubits[i]+len(error_array)//2] = 1
                elif propagated_error_name[i] == "Y":
                    propagated_error_array[gate.qubits[i]] = 1
                    propagated_error_array[gate.qubits[i]+len(error_array)//2] = 1

            return propagated_error_array

    @classmethod
    def find_next_spot(cls, current_spot: int, max_spot: int, qubit_idx: int, gates: list[Gate]) -> int:
        """Finds the next propagation spot on the current qubit.
        
        Args:
            current_spot: the gate index associated with the current spot.
            max_spot: the index associated with the last spot after the last gate affecting this qubit.
            qubit_idx: the index of the qubit we are looking at.
            gates: list of the circuit gates.
            
        Returns:
            The index of the next spot.
        """
        next_spot = current_spot+1
        for j in range(next_spot, max_spot+1): # Trying to find the next gate affecting this qubit
            if j == max_spot:
                break
            elif qubit_idx in gates[j].qubits:
                next_spot = j
                break
        return next_spot


    @classmethod
    def from_circuit(cls, circ: Circuit) -> FaultSet:
        """Generate a FaultSet from a circuit
        
        Args:
            circ: The circuit to generate faults from.
            reduce: Reduce faults by stabilizers induced by the circuit.

        Returns:
            A FaultSet containing the faults generated from the circuit.
        """ 

        num_qubits = circ.num_qubits()

        qubit_faults = [{} for _ in range(num_qubits)] # For each qubit, store 
        # an array for each point in the circuit where an error could 
        # and remember an index for this spot
        
        # Go through every gate to add propagation spots identified with the
        # index of the gate
        for i in range(len(circ.gates)):
            g = circ.gates[i]
            if len(g.qubits) == 1:
                qubit_idx = g.qubits[0]
                qubit_faults[qubit_idx][i] = {}
                qubit_faults[qubit_idx][i]['I'] = np.zeros(2*num_qubits, dtype=np.int8)
                qubit_faults[qubit_idx][i]['X'] = 0
                qubit_faults[qubit_idx][i]['Z'] = 0
                qubit_faults[qubit_idx][i]['Y'] = 0
            if len(g.qubits) == 2:
                ctrl, trgt = g.qubits
                qubit_faults[ctrl][i] = {}
                qubit_faults[ctrl][i]['I'] = np.zeros(2*num_qubits, dtype=np.int8)
                qubit_faults[ctrl][i]['X'] = 0
                qubit_faults[ctrl][i]['Z'] = 0
                qubit_faults[ctrl][i]['Y'] = 0
                qubit_faults[trgt][i] = {}
                qubit_faults[trgt][i]['I'] = np.zeros(2*num_qubits, dtype=np.int8)
                qubit_faults[trgt][i]['X'] = 0
                qubit_faults[trgt][i]['Z'] = 0
                qubit_faults[trgt][i]['Y'] = 0
        
        # Add one more spot for each qubit after the last gate affecting it
        for qb in range(num_qubits):
            last_gate_idx = max(qubit_faults[qb].keys())
            qubit_faults[qb][last_gate_idx+1] = {}

            no_error = np.zeros(2*num_qubits, dtype=np.int8)
            qubit_faults[qb][last_gate_idx+1]['I'] = no_error
            
            x_error = np.zeros(2*num_qubits, dtype=np.int8)
            x_error[qb] = 1
            qubit_faults[qb][last_gate_idx+1]['X'] = x_error
            
            z_error = np.zeros(2*num_qubits, dtype=np.int8)
            z_error[qb+num_qubits] = 1
            qubit_faults[qb][last_gate_idx+1]['Z'] = z_error
            
            y_error = np.zeros(2*num_qubits, dtype=np.int8)
            y_error[qb] = 1
            y_error[qb+num_qubits] = 1
            qubit_faults[qb][last_gate_idx+1]['Y'] = y_error

        # Now qubit_faults contains an empty array for every possible 
        # point in the circuit where an error could start to propagate  


        # Iterate through the circuit in reverse and forward propagate faults
        for i in range(len(circ.gates)-1, -1, -1):

            g = circ.gates[i]

            # print(f"Propagating through gate {i}: {g.name} on qubits {g.qubits}")

            if len(g.qubits) == 1:
                
                qubit_idx = g.qubits[0]

                for f in ["X", "Y", "Z"]:

                    # Propagate fault f
                    error = np.zeros(2*num_qubits, dtype=np.int8)
                    if f == "X":
                        error[qubit_idx] = 1
                    elif f == "Z":
                        error[qubit_idx+num_qubits] = 1
                    elif f == "Y":
                        error[qubit_idx] = 1
                        error[qubit_idx+num_qubits] = 1

                    propagated_error = cls.forward_propagate(g, error)
                    
                    # I want to know which error is on this qubit after the propagation of fault f
                    error_name = FaultSet.convert_2n_array_to_name(propagated_error, qubit_idx)
                    
                    # Now find the next spot for this qubit
                    max_spot = max(qubit_faults[qubit_idx].keys())
                    next_spot = cls.find_next_spot(i, max_spot, qubit_idx, circ.gates)

                    new_error = qubit_faults[qubit_idx][next_spot][error_name]
                    qubit_faults[qubit_idx][i][f] = new_error

            elif len(g.qubits) == 2:
                ctrl, trgt = g.qubits

                for f in ["X", "Y", "Z"]:
                          
                    for qb in [ctrl, trgt]: 
                        # Propagate fault f
                        error = np.zeros(2*num_qubits, dtype=np.int8)
                        if f == "X":
                            error[qb] = 1
                        elif f == "Z":
                            error[qb+num_qubits] = 1
                        elif f == "Y":
                            error[qb] = 1
                            error[qb+num_qubits] = 1

                        propagated_error = cls.forward_propagate(g, error) # 2n array describing the propagated error
                        # print(f"Propagated error for qubit {qb} and fault {f} after gate {i}: {propagated_error}")

                        # I want to know which error is on the ctrl qubit after the gate
                        error_name_ctrl = cls.convert_2n_array_to_name(propagated_error, ctrl)

                        # I want to know which error is on the target qubit after the gate
                        error_name_trgt = cls.convert_2n_array_to_name(propagated_error, trgt)

                        # Now find the next spot for the control qubit
                        max_spot_ctrl = max(qubit_faults[ctrl].keys())
                        next_spot_ctrl = cls.find_next_spot(i, max_spot_ctrl, ctrl, circ.gates)

                        # print(f"Next spot for control qubit {ctrl} after gate {i}: {next_spot_ctrl}")

                        # Now find the next spot for the target qubit
                        max_spot_trgt = max(qubit_faults[trgt].keys())
                        next_spot_trgt = cls.find_next_spot(i, max_spot_trgt, trgt, circ.gates)

                        # print(f"Next spot for target qubit {trgt} after gate {i}: {next_spot_ctrl}")

                        new_error = qubit_faults[ctrl][next_spot_ctrl][error_name_ctrl] ^ qubit_faults[trgt][next_spot_trgt][error_name_trgt]
                        # print(f"New error for qubit {qb} after gate {i} and fault {f}: {new_error}")
                        qubit_faults[qb][i][f] = new_error
                    
            # print(" -----------------------------")
                    
                    
        # return qubit_faults
        # Create the fault set
        a = []
        for qb in qubit_faults:
            for qb_idx, faults in qb.items():
                for fault_name, propagated_error in faults.items():
                    if fault_name != "I":
                        a.append(list(propagated_error))
        fs = cls.from_fault_array(np.array(a, dtype=np.int8)) 
        return fs
        


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

    def coset_leader(fault: np.ndarray[np.int8], generators: np.ndarray[np.int8]) -> np.ndarray[np.int8]:
        """Compute the coset leader of a fault given a set of stabilizer generators
        
        Returns:
            The symbolic representation of the coset leader of the fault.
        """
        if len(generators) == 0:
            return fault
        
        n = len(fault) // 2

        s = z3.Optimize()

        # Create symbolic variables for the coset leader [e_0, ..., e_{2n-1}]
        leader = [z3.Bool(f"e_{i}") for i in range(len(fault))]

        # Create coefficient variables for generators
        coeff = [z3.Bool(f"c_{i}") for i in range(len(generators))]

        # Compute the symbolic stabilizer combination
        g = vars_to_stab(coeff, generators)
        
        # Add constraint: leader = fault XOR g (in GF(2))
        s.add(symbolic_vector_eq(np.array(leader), symbolic_vector_add(fault.astype(bool), g)))

        # Create the weight objective: count qubits with X OR Z errors
        # For each qubit i, we have leader[i] (X part) and leader[n+i] (Z part)
        # We want to minimize the number of qubits where (leader[i] OR leader[n+i]) is True
        weight_terms = []
        for i in range(n):
            x_part = leader[i]
            z_part = leader[n + i]
            # Create symbolic OR: qubit i has an error if X part OR Z part is 1
            qubit_error = z3.Or(x_part, z_part)
            weight_terms.append(qubit_error)
    
        # Minimize the total number of qubits with errors
        s.minimize(z3.Sum(weight_terms))
    
        # Solve (always satisfiable)
        s.check()  
        m = s.model()
    
        # Extract the solution and convert to binary array
        result = np.array([bool(m[leader[i]]) for i in range(len(fault))], dtype=np.int8)
        return result
"""Synthetizing state preparation circuits for non-CSS codes."""

from __future__ import annotations

import numpy as np
import z3
import logging

from mqt.qecc.circuit_synthesis.non_css_faults import FaultSet, coset_leader, product_fault_set
from mqt.qecc.circuit_synthesis.non_css_circuits import Circuit
from .synthesis_utils import vars_to_stab, iterative_search_with_timeout, run_with_timeout


logger = logging.getLogger(__name__)

class NCSSFaultyStatePrepCircuit:
    """Represents a state preparation circuit for a non-CSS code."""

    def __init__(self, circ: Circuit, stabilizers: np.ndarray, max_errors: int) -> None:
        """Initialize a state preparation circuit.
        
        Args: 
            circ: The state preparation circuit.
            stabilizers: The list of stabilizer generators in symplectic format.
            max_errors: Macimum number of independent errors that can happen in the circuit.
        """
        self.circ = circ
        self.stabs = stabilizers
        self.num_qubits = circ.num_qubits()
        self.max_errors = max_errors
        
        self.fault_sets: list[FaultSet] = []
        self.fault_sets_unreduced: list[FaultSet] = []

    def compute_fault_set(self, num_errors: int = 1) -> FaultSet:
        """Compute the fault set of the state.
        
        Args:
            num_errors: The number of independant errors to propagate through the circuit.
            reduce: If True, reduce the fault set by the stabilizers of the code to reduce weights.
            
        Returns:
            The fault set of the state.
        """
        if num_errors == 0:
            return FaultSet(self.num_qubits)
        
        fault_sets = self.fault_sets
        fault_sets_unreduced = self.fault_sets_unreduced

        if len(fault_sets) >= num_errors:
            return fault_sets[num_errors - 1] # return cached value
        
        if num_errors <= 0:
            msg = "Cannot compute fault set for less than 1 error."
            raise ValueError(msg)
        elif num_errors == 1:
            logger.info("Computing fault set for 1 error.")
            fs = FaultSet.from_circuit(self.circ)
        else:
            logger.info(f"Computing fault set for {num_errors} errors.")
            self.compute_fault_set(num_errors - 1)
            faults = fault_sets[num_errors - 2]
            single_faults = fault_sets_unreduced[0]

            fs = product_fault_set(faults, single_faults)
            fs.remove_zero_rows()
            fs.remove_duplicates()

        fault_sets_unreduced.append(fs.copy())

        logger.info("Removing low-weight faults.")
        fs.filter_by_weight_at_least(num_errors + 1, self.stabs)
        fault_sets.append(fs)

        return fs


def gate_optimal_verification_stabilizers(
    fault_sets: list[FaultSet],
    stabs: np.ndarray[np.int8],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None
) -> list[list[np.ndarray[np.int8]]]:
    """Return verification stabilizers for the given fault sets.

    Args:
        fault_sets: List of fault sets to verify.
        stabs: The stabilizer generators to verify the fault sets.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.

    Returns:
        A list of stabilizers for each number of errors to verify the state preparation circuit.
    """
    return [
        layers[0] if layers != [] else []
        for layers in all_gate_optimal_verification_stabilizers(
            fault_sets,
            stabs,
            min_timeout,
            max_timeout,
            max_ancillas,
            return_all_solutions=False
        )
    ]

def all_gate_optimal_verification_stabilizers(
    fault_sets: list[FaultSet],
    stabs: np.ndarray[np.int8],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int| None = None,
    return_all_solutions: bool = False,
    weight_z: int = 1,
    weight_x: int = 2,
    weight_y: int = 2,
) -> list[list[list[np.ndarray[np.int8]]]]:
    """Return all equivalent verification stabilizers for the given fault sets.

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and gates.
    This is repeated for each number of independent correctable errors in the state preparation circuit.
    Thus the verificatioon circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        fault_sets: List of fault sets to verify.
        stabs: The stabilizer generators to verify the fault sets.

    Returns:
        A list of all equivalent stabilizers for each number of errors to verify the state preparation circuit.
    """

    n_layers = len(fault_sets)
    layers: list[list[list[np.ndarray[np.int8]]]] = [[] for _ in range(n_layers)]
    if max_ancillas is None:
        max_ancillas = stabs.shape[0] # by default number of stabilizer generators

    n_qubits = stabs.shape[1] // 2

    def row_cost(row: np.ndarray[np.int8]) -> int:
        """Compute the cost of a stabilizer row based on the number of X, Y, and Z terms it contains."""
        x = row[:n_qubits]
        z = row[n_qubits:]
        cost = 0
        for q in range(n_qubits):
            if x[q] == 1 and z[q] == 1:
                cost += weight_y
            elif x[q] == 1:
                cost += weight_x
            elif z[q] == 1:
                cost += weight_z
            else:
                continue
        return cost
    
    min_row_cost = int(min(row_cost(row) for row in stabs))
    max_row_cost = int(max(row_cost(row) for row in stabs))

    # Find the optimal circuit for every number of errors int the preparation circuit
    for layer in range(n_layers):
        logger.info(f"Finding verification stabilizers for {layer + 1} errors")
        faults = fault_sets[layer]

        if len(faults) == 0:
            logger.info(f"No non-trivial faults for {layer + 1} errors.")
            layers[layer] = []
            continue

        # Start with the maximal number of ancillas
        # A minimal gates solution must be achievable with these
        num_anc = max_ancillas
        min_cost = max(1, min_row_cost)
        max_cost = max_row_cost * num_anc

        logger.info(f"Finding verification stabilizers for {layer + 1} errors with cost {min_cost}..{max_cost} using {num_anc} ancillas")

        def fun(cost_budget: int) -> list[np.ndarray[np.int8]] | None:
            return verification_stabilizers(faults, stabs, num_anc, cost_budget, weight_x=weight_x, weight_y=weight_y, weight_z=weight_z)
        
        res = iterative_search_with_timeout(fun, min_cost, max_cost, min_timeout, max_timeout)

        if res is not None:
            measurements, curr_cost = res
        else:
            measurements = None

        if measurements is None:
            logger.info(f"No verification stabilizers found for {layer + 1} errors")
            return []  # No solution found
        
        logger.info(f"Found verification stabilizers for {layer + 1} errors with {curr_cost} cost.")
        # If any measurements are unused we can reduce the number of ancillas at least by that
        measurements = [m for m in measurements if np.any(m)]
        num_anc = len(measurements)
        # Iterate backwards to find the minimal number of cnots
        logger.info(f"Finding minimal number of CNOTs for {layer + 1} errors")

        def search_cost(cost_budget: int) -> list[np.ndarray[np.int8]] | None:
            return verification_stabilizers(faults, stabs, num_anc, cost_budget, weight_x=weight_x, weight_y=weight_y, weight_z=weight_z)
        
        while curr_cost - 1 > 0:
            logger.info(f"Trying cost {curr_cost-1}")
            cost_opt = run_with_timeout(search_cost, curr_cost-1, timeout=max_timeout)

            if cost_opt and not isinstance(cost_opt, str):
                curr_cost -= 1
                measurements = cost_opt
            else: 
                break
        
        logger.info(f"Minimal cost for {layer+1} errors is: {curr_cost}")

        # If the cost is minimal, we can reduce the number of ancillas
        logger.info(f"Finding minimal number of ancillas for {layer + 1} errors")
        while num_anc - 1 > 0:
            logger.info(f"Trying {num_anc - 1} ancillas")

            def search_anc(num_anc: int) -> list[np.ndarray[np.int8]] | None:
                return verification_stabilizers(faults, stabs, num_anc, curr_cost, weight_x=weight_x, weight_y=weight_y, weight_z=weight_z)
            
            anc_opt = run_with_timeout(search_anc, num_anc - 1, timeout=max_timeout)

            if anc_opt and not isinstance(anc_opt, str):
                num_anc -= 1
                measurements = anc_opt
            else:
                break
            
        logger.info(f"Minimal number of ancillas for {layer + 1} errors is: {num_anc}")

        if not return_all_solutions:
            layers[layer] = [measurements]
        else:
            all_stabs = all_verification_stabilizers(faults, stabs, num_anc, curr_cost, return_all_solutions=True, weight_x=weight_x, weight_y=weight_y, weight_z=weight_z)
            if all_stabs:
                layers[layer] = all_stabs
                logger.info(f"Found {len(layers[layer])} equivalent solutions for {layer} errors")
    return layers


def verification_stabilizers(
        fault_set: FaultSet,
        stabilizers: np.ndarray[np.int8],
        num_anc: int,
        max_cost: int,
        weight_z: int = 1,
        weight_x: int = 2,
        weight_y: int = 2
) -> list[np.ndarray[np.int8]] | None:
    """Return a set of stabilizers detecting all errors in `fault_set`using at most `num_anc`ancillas and at most `num_gates` gates.
    
    Args:
        fault_set: The set of errors to verify.
        stabilizers: Stabilizer generators of the stabilizers measured.
        num_anc: The maximum number of ancilla qubits to use.
        num_gates: The maximum number of gates to use.
    """
    solutions = all_verification_stabilizers(fault_set, stabilizers, num_anc, max_cost, return_all_solutions=False, weight_z=weight_z, weight_x=weight_x, weight_y=weight_y)
    if solutions is None:
        return None
    return solutions[0]


def all_verification_stabilizers(
        fault_set: FaultSet,
        stabilizers: np.ndarray[np.int8],
        num_anc: int,
        max_cost: int,
        return_all_solutions: bool = False,
        weight_z: int = 1,
        weight_x: int = 2,
        weight_y: int = 2
) -> list[list[np.ndarray[np.int8]]] | None:
    """Return a list of verification stabilizers for independant errors in the state preparation circuit using z3.
    
    Args:
        fault_set: The set of errors to verify.
        stabilizers: Stabilizer generators of the stabilizers measured shape (n_gens, 2*n_qubits) as [X|Z].
        num_anc: The maximum number of ancilla qubits to use.
        max_cost: The maximum weighted gate cost.
        return_all_solutions: If True, return all solutions. Otherwise return the first solution found
        weight_z: Cost for a Z term on one qubit in a measured stabilizer.
        weight_x: Cost for an X term.
        weight_y: Cost for a Y term.
    """

    if fault_set.faults.shape[1] != stabilizers.shape[1]:
        msg = "Fault set and stabilizers must have the same number of qubits."
        raise ValueError(msg)
    
    # Check if fault set can be verified, i.e. every fault can be detected by at least one measurement
    n = fault_set.faults.shape[1] // 2
    fx = fault_set.faults[:, :n]
    fz = fault_set.faults[:, n:]
    sx = stabilizers[:, :n]
    sz = stabilizers[:, n:]
    
    # Symplectic product: fx @ sz^T + fz @ sx^T (mod 2)
    anticommutes = (fx @ sz.T + fz @ sx.T) % 2
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

    logger.info(f"Adding constraints to the solver")

    solver = z3.Solver()
    # Assert that each error is detected
    solver.add(z3.And([
        z3.PbGe([(odd_overlap(measurement, error), 1) 
                for measurement in measurement_stabs], 1)
        for error in fault_set
    ]))

    weighted_terms: list[tuple[z3.BoolRef, int]] = []
    for measurement in measurement_stabs:
        for q in range(n_qubits):
            xq = measurement[q]
            zq = measurement[n_qubits + q]

            is_z = z3.And(z3.Not(xq), zq)
            is_x = z3.And(xq, z3.Not(zq))
            is_y = z3.And(xq, zq)

            if weight_z > 0:
                weighted_terms.append((is_z, weight_z))
            if weight_x > 0:
                weighted_terms.append((is_x, weight_x))
            if weight_y > 0:
                weighted_terms.append((is_y, weight_y))

    solver.add(z3.PbLe(weighted_terms, max_cost))

    logger.info(f"Starting search for verification stabilizers")
    print(f"Starting search for verification stabilizers")

    solutions = []
    while solver.check() == z3.sat:
        model = solver.model()
        # Extract stabilizer measurements from model
        actual_measurements = []
        for m in measurement_vars:
            v = np.zeros(2*n_qubits, dtype=np.int8)
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
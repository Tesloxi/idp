from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.non_css_circuits import *
from mqt.qecc.circuit_synthesis.non_css_faults import *

@pytest.fixture
def stabilizer_matrix():
    """Fixture for a sample stabilizer matrix."""
    return np.array([[1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 1]], dtype=np.int8) # XZX, XXZ, IYY

@pytest.fixture
def empty_stabilizer_matrix():
    """Fixture for an empty stabilizer matrix."""
    return np.zeros((0, 6), dtype=np.int8)

def test_add_fault():
    """Test adding faults to the fault set."""
    fault_set = FaultSet(num_qubits=3)

    # Add a fault
    fault_set.add_fault(np.array([1, 0, 1, 0, 1, 0], dtype=np.int8))
    assert np.array_equal(fault_set.to_array(), np.array([[1, 0, 1, 0, 1, 0]], dtype=np.int8))

    # Add another fault
    fault_set.add_fault(np.array([1, 0, 0, 1, 1, 1], dtype=np.int8))
    assert np.array_equal(fault_set.to_array(), np.array([[1, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1]], dtype=np.int8)) , (
        "Second fault was not added correctly"
    )

def test_add_fault_invalid_length():
    """Test adding a fault with an invalid length"""
    fault_set = FaultSet(num_qubits=3)

    # Attempt to add a fault with incorrect length
    with pytest.raises(ValueError, match=r"Fault must have length 6."):
        fault_set.add_fault(np.array([1, 0, 0, 0], dtype=np.int8))

def test_combine_fault_sets():
    """Test combining two fault sets."""
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 1, 0, 0, 0, 1], dtype=np.int8))

    fault_set_2 = FaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 1, 1, 0, 1, 0], dtype=np.int8))

    # Combine the fault sets
    combined_fault_set = fault_set_1.combine(fault_set_2)
    expected_faults = np.array([[1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 0]], dtype=np.int8)
    assert combined_fault_set.to_set() == set(map(tuple, expected_faults)), "Fault sets were not combined correctly."

def test_combine_fault_sets_invalid():
    """Test combining fault sets with different number of qubits."""
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_2 = FaultSet(num_qubits=4)

    # Attempt to combine fault sets with different numbers of qubits
    with pytest.raises(ValueError, match=r"Fault sets must have the same number of qubits to combine."):
        fault_set_1.combine(fault_set_2)

def test_from_fault_array():
    """Test creating a fault set from a numpy array."""
    fault_array = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0]], dtype=np.int8)
    fault_set = FaultSet.from_fault_array(fault_array)

    # Convert the fault set to an array
    result_array = fault_set.to_array()

    # Check that the rows in the result match the expected rows, regardless of order
    assert set(map(tuple, result_array)) == set(map(tuple, fault_array)), "Fault set was not created correctly from the array."

@pytest.mark.parametrize(
        ("stabs_fixture", "initial_faults", "expected_faults"),
        [
            # Test case: Remove equivalent faults
            ("stabilizer_matrix", [[1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 1, 1]], []),
            # Test case: Fault reduced to coset representative
            ("stabilizer_matrix", [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0]], [[1, 0, 0, 0, 0, 0]]),
            # Test case: Empty stabilizer matrix 
            ("empty_stabilizer_matrix", [[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]], [[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]),
            # Test case: No reduction
            ("stabilizer_matrix", [[0, 0, 0, 1, 0, 0]], [[0, 0, 0, 1, 0, 0]])
        ]
)

def test_remove_equivalent(request, stabs_fixture, initial_faults, expected_faults):
    """Test removing equivalent faults with respect to a stabilizer group."""
    # Use the fixture dinamically
    stabs = request.getfixturevalue(stabs_fixture)

    #Initialize the fault set
    fault_set = FaultSet(num_qubits=3)
    for fault in initial_faults:
        fault_set.add_fault(np.array(fault, dtype=np.int8))

    # Remove equivalent faults
    fault_set.remove_equivalent(stabs)

    # Check the result
    assert fault_set.to_set() == set(map(tuple, expected_faults)), (
        "Fault set was not reduced to unique coset representatives correctly."
    )

def test_filter_by_weight_basic():
    """Test filtering faults by weeight with a simple """
    stabs = np.array([[1, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=np.int8) # Stabilizers: XZI, IZZ
    fault_set = FaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 1, 0, 0, 0, 0], dtype=np.int8))
    fault_set.add_fault(np.array([1, 1, 0, 0, 1, 0], dtype=np.int8))
    fault_set.add_fault(np.array([1, 0, 0, 1, 0, 0], dtype=np.int8))
    fault_set.add_fault(np.array([0, 1, 0, 0, 0, 0], dtype=np.int8))

    # Filter faults with weight >= 2
    fault_set.filter_by_weight_at_least(2, stabs)

    # Expected faults after filtering
    expected_faults = FaultSet(num_qubits=3)
    expected_faults.add_fault(np.array([1, 1, 0, 0, 0, 0], dtype=np.int8))
    expected_faults.add_fault(np.array([1, 1, 0, 0, 1, 0], dtype=np.int8))

    assert np.array_equal(fault_set.faults, expected_faults.faults), "Faults were not filtered correctly by weight."

def test_filter_by_weight_empty_stabilizer():
    """Test filtering with an empty stabilizer group"""
    stabs = np.zeros((0, 6), dtype=np.int8) # Empty stabilizer group
    fault_set = FaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 0, 0, 0, 1], dtype=np.int8))
    fault_set.add_fault(np.array([0, 0, 1, 0, 1, 0], dtype=np.int8))

    # Filter faults with weight >= 2
    fault_set.filter_by_weight_at_least(2, stabs)

    # Expected faults after filtering
    expected_faults = FaultSet.from_fault_array(
        np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1, 0],
            ],
            dtype=np.int8
        )
    )

    assert fault_set == expected_faults, "Faults should remain unchanged when the stabilizer group is empty."

def test_filter_by_weight_complex():
    """Test filtering by weight with a complex stabilizer group."""
    stabs = np.array([
        [1, 1, 0, 0, 0, 0, 0,      0, 0, 1, 1, 0, 0, 0], # XXZZIII
        [1, 0, 0, 0, 1, 0, 1,      0, 0, 1, 0, 0, 0, 0], # XIZIXIX
        [0, 0, 1, 1, 0, 1, 0,      0, 0, 1, 0, 1, 1, 0], # IIYXZYI
    ], dtype=np.int8)

    fault_set = FaultSet(num_qubits=7)
    fault_set.add_fault(np.array([1, 1, 1, 1, 1, 1, 1,      1, 1, 1, 1, 1, 1, 1], dtype=np.int8)) # YYYYYYY
    fault_set.add_fault(np.array([1, 0, 0, 0, 0, 0, 0,      0, 1, 0, 0, 0, 0, 0], dtype=np.int8)) # XZ
    fault_set.add_fault(np.array([0, 1, 1, 1, 0, 0, 0,      0, 0, 0, 0, 0, 1, 1], dtype=np.int8)) # IXXXIZZ
    fault_set.add_fault(np.array([0, 0, 0, 0, 1, 0, 0,      0, 0, 0, 0, 0, 1, 1], dtype=np.int8)) # IIIIXZZ

    fault_set.filter_by_weight_at_least(2, stabs)

    expected_faults = FaultSet(num_qubits=7)
    expected_faults.add_fault(np.array([1, 0, 0, 0, 0, 0, 0,      0, 1, 0, 0, 0, 0, 0], dtype=np.int8)) 
    expected_faults.add_fault(np.array([0, 0, 0, 0, 1, 0, 0,      0, 0, 0, 0, 0, 1, 1], dtype=np.int8))

    assert stabilizer_equivalent(fault_set, expected_faults, stabs), (
        "Faults were not filtered correctly by weight with a complex stabilizer group."
    )

def test_stabilizer_equivalent_identical_fault_sets():
    """Test equivalence of two identical fault sets."""
    stabs = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=np.int8)
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 0, 1, 1, 0], dtype=np.int8))
    fault_set_1.add_fault(np.array([0, 0, 1, 1, 1, 1], dtype=np.int8))
    
    fault_set_2 = FaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([1, 0, 0, 1, 1, 0], dtype=np.int8))
    fault_set_2.add_fault(np.array([0, 0, 1, 1, 1, 1], dtype=np.int8))

    # Check equivalence
    assert stabilizer_equivalent(fault_set_1, fault_set_2, stabs), "Identical fault sets should be equivalent."

def test_stabilizer_equivalent_different_fault_sets():
    """Test non-equivalence of two different fault sets."""
    stabs = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=np.int8)
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 0, 0, 1, 0], dtype=np.int8))
    
    fault_set_2 = FaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 0, 1, 0, 0, 1], dtype=np.int8))

    # Check equivalence
    assert not stabilizer_equivalent(fault_set_1, fault_set_2, stabs), "Different fault sets should not be equivalent."

def test_stabilizer_equivalent_equivalent_fault_sets():
    """Test equivalence of two different fault sets that are equivalent under the stabilizer group."""
    stabs = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=np.int8)
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 1, 0, 1, 0], dtype=np.int8))
    
    fault_set_2 = FaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 0, 0, 0, 0, 0], dtype=np.int8))

    # Check equivalence
    assert stabilizer_equivalent(fault_set_1, fault_set_2, stabs), "Fault sets equivalent under the stabilizer group should be equivalent."

def test_stabilizer_equivalent_different_num_qubits():
    """Test that fault sets with different numbers of qubits raise an error."""
    stabs = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]], dtype=np.int8)
    fault_set_1 = FaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 0, 0, 1, 0], dtype=np.int8))
    
    fault_set_2 = FaultSet(num_qubits=4)
    fault_set_2.add_fault(np.array([0, 0, 1, 0, 0, 1, 0, 1], dtype=np.int8))

    # Check for ValueError
    with pytest.raises(ValueError, match=r"Fault sets must have the same number of qubits to compare."):
        stabilizer_equivalent(fault_set_1, fault_set_2, stabs)

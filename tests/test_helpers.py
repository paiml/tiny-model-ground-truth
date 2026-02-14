"""Property-based tests for parity checker helper functions.

These tests verify the internal logic of the parity checker using
hypothesis-generated inputs. They do NOT require apr or model files.

Test categories:
  - property_based: hypothesis property-based tests (fuzz)
  - negative: expected_failure / should_fail assertions
  - unit: known input/output pairs
"""

import pytest
from helpers import count_mismatches
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.property_based
@given(st.lists(st.integers(0, 50000), max_size=64))
@settings(max_examples=100)
def test_count_mismatches_identity(tokens):
    """Identical sequences should produce 0 mismatches."""
    assert count_mismatches(tokens, tokens) == 0


@pytest.mark.property_based
@given(
    st.lists(st.integers(0, 50000), min_size=1, max_size=64),
    st.lists(st.integers(0, 50000), min_size=1, max_size=64),
)
@settings(max_examples=100)
def test_count_mismatches_symmetric(a, b):
    """Mismatch count should be symmetric: f(a,b) == f(b,a)."""
    assert count_mismatches(a, b) == count_mismatches(b, a)


@pytest.mark.property_based
@given(
    st.lists(st.integers(0, 50000), min_size=1, max_size=64),
    st.lists(st.integers(0, 50000), min_size=1, max_size=64),
)
@settings(max_examples=100)
def test_count_mismatches_upper_bound(a, b):
    """Mismatch count should never exceed max(len(a), len(b))."""
    m = count_mismatches(a, b)
    assert m <= max(len(a), len(b))


@pytest.mark.property_based
@given(st.lists(st.integers(0, 50000), min_size=1, max_size=32))
@settings(max_examples=100)
def test_count_mismatches_empty_vs_nonempty(tokens):
    """Empty vs non-empty should equal len(non-empty)."""
    assert count_mismatches([], tokens) == len(tokens)
    assert count_mismatches(tokens, []) == len(tokens)


def test_count_mismatches_known_values():
    """Known input/output pairs for count_mismatches."""
    assert count_mismatches([1, 2, 3], [1, 2, 3]) == 0
    assert count_mismatches([1, 2, 3], [1, 2, 4]) == 1
    assert count_mismatches([1, 2, 3], [4, 5, 6]) == 3
    assert count_mismatches([1, 2], [1, 2, 3]) == 1
    assert count_mismatches([], []) == 0


def test_count_mismatches_should_fail_on_divergent():
    """Negative test (expected_failure scenario): fully divergent sequences should have max mismatches."""
    a = list(range(32))
    b = list(range(32, 64))
    assert count_mismatches(a, b) == 32


def test_should_fail_mismatches_exceed_threshold():
    """Negative test: should_fail when mismatches exceed Int4 threshold of 5/32."""
    oracle = list(range(32))
    divergent = list(range(10, 42))  # 32 mismatches
    m = count_mismatches(oracle, divergent)
    # This SHOULD produce a threshold violation (m > 5)
    assert m > 5, f"Expected threshold violation but got {m} mismatches"

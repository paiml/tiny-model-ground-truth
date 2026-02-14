"""Property-based tests using hypothesis (PMAT-002).

Fuzzes count_mismatches (both helpers.py and parity_check.py versions),
Result class, and format_ticket with random inputs.
"""

import pytest
from helpers import count_mismatches as helpers_count_mismatches
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.parity_check import (
    Result,
    format_ticket,
)
from scripts.parity_check import (
    count_mismatches as pc_count_mismatches,
)

# ── Strategies ──────────────────────────────────────────────────

# Token IDs (realistic range for small models)
token_lists = st.lists(st.integers(min_value=0, max_value=200000), max_size=100)


# ── count_mismatches properties ─────────────────────────────────


@pytest.mark.property_based
@given(a=token_lists, b=token_lists)
def test_count_mismatches_non_negative(a, b):
    """Mismatch count is always >= 0."""
    assert helpers_count_mismatches(a, b) >= 0
    assert pc_count_mismatches(a, b) >= 0


@pytest.mark.property_based
@given(a=token_lists)
def test_count_mismatches_identity(a):
    """Identical lists have 0 mismatches."""
    assert helpers_count_mismatches(a, a) == 0
    assert pc_count_mismatches(a, a) == 0


@pytest.mark.property_based
@given(a=token_lists, b=token_lists)
def test_count_mismatches_symmetric(a, b):
    """count_mismatches(a, b) == count_mismatches(b, a)."""
    assert helpers_count_mismatches(a, b) == helpers_count_mismatches(b, a)
    assert pc_count_mismatches(a, b) == pc_count_mismatches(b, a)


@pytest.mark.property_based
@given(a=token_lists, b=token_lists)
def test_count_mismatches_upper_bound(a, b):
    """Mismatches never exceed max(len(a), len(b))."""
    bound = max(len(a), len(b))
    assert helpers_count_mismatches(a, b) <= bound
    assert pc_count_mismatches(a, b) <= bound


@pytest.mark.property_based
@given(a=token_lists, b=token_lists)
def test_count_mismatches_length_diff_lower_bound(a, b):
    """Mismatches >= abs(len(a) - len(b)) due to length difference."""
    diff = abs(len(a) - len(b))
    assert helpers_count_mismatches(a, b) >= diff
    assert pc_count_mismatches(a, b) >= diff


@pytest.mark.property_based
@given(a=token_lists)
def test_count_mismatches_empty_vs_list(a):
    """Mismatches against empty list equals len(a)."""
    assert helpers_count_mismatches(a, []) == len(a)
    assert helpers_count_mismatches([], a) == len(a)


@pytest.mark.property_based
@given(a=token_lists, b=token_lists)
def test_count_mismatches_implementations_agree(a, b):
    """helpers.py and parity_check.py implementations produce same result."""
    assert helpers_count_mismatches(a, b) == pc_count_mismatches(a, b)


@pytest.mark.property_based
@given(
    n=st.integers(min_value=0, max_value=50),
    data=st.data(),
)
@settings(max_examples=50)
def test_count_mismatches_triangle_inequality(n, data):
    """Triangle inequality: mismatches(a,c) <= mismatches(a,b) + mismatches(b,c)."""
    # Only holds when lists have same length
    fixed_len_lists = st.lists(st.integers(min_value=0, max_value=200000), min_size=n, max_size=n)
    a = data.draw(fixed_len_lists)
    b = data.draw(fixed_len_lists)
    c = data.draw(fixed_len_lists)
    assert helpers_count_mismatches(a, c) <= (
        helpers_count_mismatches(a, b) + helpers_count_mismatches(b, c)
    )


# ── Result class properties ─────────────────────────────────────


@pytest.mark.property_based
@given(name=st.text(min_size=1, max_size=100))
def test_result_init_always_fails(name):
    """New Result starts in failed state."""
    r = Result(name)
    assert r.name == name
    assert r.passed is False
    assert r.error == ""


@pytest.mark.property_based
@given(name=st.text(min_size=1, max_size=50), details=st.text(max_size=200))
def test_result_pass_sets_passed(name, details):
    """pass_() always sets passed=True."""
    r = Result(name)
    r.pass_(details)
    assert r.passed is True
    assert r.details == details


@pytest.mark.property_based
@given(
    name=st.text(min_size=1, max_size=50),
    error=st.text(min_size=1, max_size=200),
    details=st.text(max_size=200),
)
def test_result_fail_sets_failed(name, error, details):
    """fail() always sets passed=False."""
    r = Result(name)
    r.fail(error, details)
    assert r.passed is False
    assert r.error == error
    assert r.details == details


@pytest.mark.property_based
@given(
    name=st.text(min_size=1, max_size=50),
    details=st.text(max_size=200),
    error=st.text(min_size=1, max_size=200),
)
def test_result_fail_after_pass_overrides(name, details, error):
    """fail() after pass_() must set passed=False (no sticky pass)."""
    r = Result(name)
    r.pass_(details)
    assert r.passed is True
    r.fail(error)
    assert r.passed is False


@pytest.mark.property_based
@given(
    name=st.text(min_size=1, max_size=50),
    error=st.text(min_size=1, max_size=200),
    details=st.text(max_size=200),
)
def test_result_pass_after_fail_overrides(name, error, details):
    """pass_() after fail() must set passed=True (no sticky fail)."""
    r = Result(name)
    r.fail(error)
    assert r.passed is False
    r.pass_(details)
    assert r.passed is True


# ── format_ticket properties ────────────────────────────────────


@pytest.mark.property_based
@given(
    name=st.text(min_size=1, max_size=50, alphabet=st.characters(blacklist_categories=("Cs",))),
    error=st.text(min_size=1, max_size=200, alphabet=st.characters(blacklist_categories=("Cs",))),
)
def test_format_ticket_contains_name_and_error(name, error):
    """format_ticket output always contains the check name and error."""
    r = Result(name)
    r.fail(error)
    ticket = format_ticket([r])
    assert name in ticket
    assert error in ticket


@pytest.mark.property_based
@given(
    names=st.lists(
        st.text(min_size=1, max_size=30, alphabet=st.characters(blacklist_categories=("Cs",))),
        min_size=1,
        max_size=10,
    ),
)
def test_format_ticket_includes_all_failures(names):
    """Every failed result appears in the ticket."""
    results = []
    for n in names:
        r = Result(n)
        r.fail(f"error in {n}")
        results.append(r)
    ticket = format_ticket(results)
    for n in names:
        assert n in ticket


@pytest.mark.property_based
@given(n=st.integers(min_value=0, max_value=20))
def test_format_ticket_failure_count_matches(n):
    """format_ticket reports correct count of failures passed to it."""
    results = []
    for i in range(n):
        r = Result(f"check/{i}")
        r.fail(f"error {i}")
        results.append(r)
    ticket = format_ticket(results)
    assert f"{n} failure(s)" in ticket

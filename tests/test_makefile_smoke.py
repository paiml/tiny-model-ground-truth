"""Makefile target smoke tests (PMAT-005).

Validates that:
1. All .PHONY targets exist and are parseable
2. CLAUDE.md command table stays in sync with Makefile
3. Key targets are present
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
MAKEFILE = ROOT / "Makefile"
CLAUDE_MD = ROOT / "CLAUDE.md"

# ── Parse helpers ───────────────────────────────────────────────


def _parse_phony_targets() -> set[str]:
    """Extract all .PHONY targets from the Makefile."""
    content = MAKEFILE.read_text()
    targets = set()
    for line in content.splitlines():
        if line.startswith(".PHONY:"):
            # .PHONY: target1 target2 ...
            parts = line.split(":", 1)[1].strip().split()
            targets.update(parts)
    return targets


def _parse_makefile_targets() -> set[str]:
    """Extract all defined targets (lines matching 'name:')."""
    content = MAKEFILE.read_text()
    targets = set()
    for line in content.splitlines():
        # Match lines like "target:" or "target: dep1 dep2"
        # Skip comments, variable assignments, indented lines
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:", line)
        if m:
            targets.add(m.group(1))
    return targets


def _parse_claude_md_commands() -> set[str]:
    """Extract make commands from the CLAUDE.md commands table."""
    content = CLAUDE_MD.read_text()
    commands = set()
    for line in content.splitlines():
        # Match lines like "| `make target` | description |"
        m = re.search(r"`make\s+([a-zA-Z][a-zA-Z0-9_-]*)`", line)
        if m:
            commands.add(m.group(1))
    return commands


# ── Tests ───────────────────────────────────────────────────────


def test_makefile_exists():
    """Makefile must exist at project root."""
    assert MAKEFILE.exists()


def test_claude_md_exists():
    """CLAUDE.md must exist at project root."""
    assert CLAUDE_MD.exists()


def test_phony_targets_not_empty():
    """Makefile must declare .PHONY targets."""
    targets = _parse_phony_targets()
    assert len(targets) > 0


def test_all_phony_targets_are_defined():
    """Every .PHONY target must have a corresponding rule."""
    phony = _parse_phony_targets()
    defined = _parse_makefile_targets()
    # Some targets may be defined as dependencies (e.g. 'ci: pull convert check')
    # so they appear in .PHONY but their rule is the dependency line
    missing = phony - defined
    assert not missing, f".PHONY targets without rules: {missing}"


ESSENTIAL_TARGETS = [
    "test",
    "coverage",
    "check",
    "oracle",
    "pull",
    "convert",
    "clean",
    "ci",
    "ticket",
    "lint",
]


@pytest.mark.parametrize("target", ESSENTIAL_TARGETS)
def test_essential_target_exists(target):
    """Essential targets must be defined in the Makefile."""
    defined = _parse_makefile_targets()
    assert target in defined, f"Missing essential target: {target}"


def test_claude_md_commands_are_in_makefile():
    """Every command listed in CLAUDE.md must exist in the Makefile."""
    claude_cmds = _parse_claude_md_commands()
    makefile_targets = _parse_makefile_targets()
    missing = claude_cmds - makefile_targets
    assert not missing, (
        f"CLAUDE.md references targets not in Makefile: {missing}. "
        f"Update CLAUDE.md or add the target to Makefile."
    )


def test_key_makefile_targets_in_claude_md():
    """Key Makefile targets should be documented in CLAUDE.md."""
    # Not every internal target needs to be documented, but the user-facing ones should
    user_facing = {"test", "coverage", "check", "oracle", "pull", "convert", "clean", "ci", "ticket"}
    claude_cmds = _parse_claude_md_commands()
    missing = user_facing - claude_cmds
    assert not missing, (
        f"User-facing targets not documented in CLAUDE.md: {missing}. "
        f"Add `make {next(iter(missing))}` to the commands table."
    )


def test_test_target_excludes_requires_apr():
    """make test must use -m 'not requires_apr' to skip integration tests."""
    content = MAKEFILE.read_text()
    # Find the test target's recipe
    in_test = False
    for line in content.splitlines():
        if line.startswith("test:"):
            in_test = True
            continue
        if in_test:
            if line.startswith("\t"):
                assert "not requires_apr" in line, (
                    "make test must filter out requires_apr tests"
                )
                break
            else:
                in_test = False


def test_coverage_target_uses_cov_flag():
    """make coverage must use --cov for coverage measurement."""
    content = MAKEFILE.read_text()
    in_coverage = False
    for line in content.splitlines():
        if line.startswith("coverage:"):
            in_coverage = True
            continue
        if in_coverage:
            if line.startswith("\t"):
                assert "--cov" in line, "make coverage must use --cov flag"
                break
            else:
                in_coverage = False


def test_no_duplicate_phony_targets():
    """No target should appear twice in .PHONY declarations."""
    content = MAKEFILE.read_text()
    all_targets = []
    for line in content.splitlines():
        if line.startswith(".PHONY:"):
            parts = line.split(":", 1)[1].strip().split()
            all_targets.extend(parts)
    seen = set()
    dupes = set()
    for t in all_targets:
        if t in seen:
            dupes.add(t)
        seen.add(t)
    assert not dupes, f"Duplicate .PHONY targets: {dupes}"

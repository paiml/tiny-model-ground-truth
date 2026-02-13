# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Replaced ruchy test theater with Python parity checker (`scripts/parity_check.py`)
- Parity checker shells out to `apr run --json` and compares against oracle
- 59 checks across 5 suites (canary, token, drift, roundtrip, ppl)
- `make check` replaces `make test`; `make ticket` generates GitHub issue markdown
- Removed 7 ruchy test files (never ran real inference)

### Found
- 9 issues filed against aprender/realizar (#231-#237, #239)
- Current blocker: realizar loader reads Q8/Q4 bytes as f32 (#239)
- 0/59 checks passing

## [0.1.0] - 2026-02-13

### Added
- Initial repository with Popperian falsification framework
- Oracle golden JSON files for 3 models (SmolLM-135M, Qwen2-0.5B, GPT-2) x 4 prompts
- Pre-registered falsifiable claims in CLAIMS.md
- Design decisions documented as ADRs
- CI configuration (GitHub Actions with weekly cron)

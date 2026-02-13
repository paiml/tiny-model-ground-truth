# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-13

### Added
- Initial repository with Popperian falsification framework
- Oracle golden JSON files for 3 models (SmolLM-135M, Qwen2-0.5B, GPT-2) x 4 prompts
- 6 ruchy test suites: canary, token parity, quant drift, format roundtrip, runtime parity, perplexity
- 75 total test cases, all passing
- CI configuration (GitHub Actions with weekly cron)
- Pre-registered falsifiable claims in CLAIMS.md
- Design decisions documented as ADRs in CLAIMS.md

### Claims Registered
- Claim 1: Int8 canary parity (exact text match)
- Claim 2: Int4 token parity (≤5/32 mismatches)
- Claim 3: Int8 token parity (≤3/32 mismatches)
- Claim 4: Quantization drift ordering (Int8 ≤ Int4 + 1)
- Claim 5: Format roundtrip losslessness (exact token match)
- Claim 6: Cross-runtime parity (apr vs llama.cpp, exact text)
- Claim 7: Perplexity bounds (model-specific ceilings)
- Claim 8: Perplexity drift bound (|Int4 - Int8| < 0.5)

# Contributing

## Adding a New Model

1. Add the model to `MODELS` dict in `scripts/gen_oracle.py`
2. Run `uv run python scripts/gen_oracle.py --model <slug>`
3. Add import/export lines to `Makefile` convert target
4. Add test cases to each `.ruchy` test file
5. Update `CLAIMS.md` with the new sample sizes
6. Update `README.md` parity matrix

## Adding a New Prompt

1. Create `prompts/<name>.txt` with the prompt text
2. Regenerate all oracles: `uv run python scripts/gen_oracle.py --all`
3. Commit the new oracle JSON files
4. Tests automatically pick up new prompts from oracle directory

## Modifying Tolerances

Any change to tolerance thresholds must be documented:
1. Update `CLAIMS.md` with new threshold and justification
2. Update `CLAUDE.md` tolerance table
3. Update `README.md` tolerance table
4. Add entry to `CHANGELOG.md` explaining the deviation from pre-registration

## Test Coverage

All test files must include:
- Property-based invariant tests (structural properties)
- Boundary tests (threshold edge cases)
- Regression tests (canary against oracle)

## Commit Format

```
type: description (Refs TMGT-XXX)
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `ci`

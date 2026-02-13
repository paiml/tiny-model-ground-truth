"""Benchmark configuration for parity check suites.

Criterion-style configuration for reproducible benchmarking.
sample_size = 59 checks (exhaustive cross-product).
measurement_time = 120s timeout per check.
warm_up_time = 0s (deterministic, no warmup needed).
confidence_level = 0.95 for timing measurements.
"""

# Benchmark parameters (criterion-compatible naming)
SAMPLE_SIZE = 59
MEASUREMENT_TIME = 120  # seconds
WARM_UP_TIME = 0  # deterministic, no warmup
CONFIDENCE_LEVEL = 0.95
NOISE_THRESHOLD = 0.01  # 1% timing noise threshold
NRESAMPLES = 100  # bootstrap resamples for timing CI

SUITES = {
    "canary": {"checks": 12, "threshold": "exact match"},
    "token": {"checks": 24, "threshold": "<=5/32 (Int4), <=3/32 (Int8)"},
    "drift": {"checks": 12, "threshold": "Int8 <= Int4 + 1"},
    "roundtrip": {"checks": 6, "threshold": "exact token match"},
    "ppl": {"checks": 9, "threshold": "model-specific ceiling"},
}

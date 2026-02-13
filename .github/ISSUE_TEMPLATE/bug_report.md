---
name: Bug report
about: Report a parity check failure or oracle discrepancy
title: ''
labels: bug
assignees: ''
---

**Which claim failed?**
Reference a claim number from CLAIMS.md (1-8).

**Failure details**
```
Paste parity_check.py output here
```

**Environment**
- apr version: `apr --version`
- Python version: `python --version`
- OS:

**Reproduction**
```bash
make pull && make convert && make check
```

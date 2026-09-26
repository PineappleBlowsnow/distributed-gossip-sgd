# Packaging validation

Date: September 26, 2026. This page distinguishes checks performed during packaging from historical experiments.

## Completed

- Parsed all four packaged Python files successfully with Python's AST parser.
- Verified SHA-256 preservation of 26 original files: submitted source, 40-step summary, and 24 short-run CSVs.
- Checked that each historical set has exactly the 24 distinct topology / mode / `nu` combinations.
- Checked all 24 short-run trajectories against their summaries, including final accuracy, disagreement, sample counts and the recorded step sequence.
- Compared the source AST: graph builders, model, flatten/unflatten, communication functions, disagreement and accuracy are unchanged apart from function documentation. Worker/main edits are documented in [provenance](provenance.md).
- Exercised the dependency-free graph functions for four-node ring/star/complete/grid and nine-node grid; checked vertex bounds, edge uniqueness and symmetric neighbor lists.
- Checked Markdown relative links and scanned the package for home-directory paths, email addresses, private keys and common credential patterns. No matches were found. This is a limited pattern scan, not a guarantee covering every possible secret format.

The result checker is reproducible without third-party packages:

```bash
python verify_artifacts.py
```

Output:

```text
PASS: Python syntax; 26 preserved files; 24 short-run trajectories / summaries; 24 separate long-run records
This verifies archived artifacts, not distributed execution or fresh MNIST results.
```

## Not executed during packaging

The available Python runtime did not contain PyTorch or Torchvision. No large dependency installation, MNIST download, distributed smoke run, fresh training or timing benchmark was performed.

`smoke_test.py` is an added, syntax-checked four-process test entry point. It checks the actual Gloo communication functions, a known synchronous update, pairwise averaging, sum preservation and disagreement on small vectors. It has **not yet been run** in this preparation environment. The `--synthetic` training command in the README likewise remains unexecuted.

The 40-step and 200-step outputs are historical records. They are not September 2026 test results and are not evidence that the current dependencies reproduce those results exactly.

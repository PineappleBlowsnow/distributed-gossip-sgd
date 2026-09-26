# Provenance and packaging changes

## Original materials

- `archive/gossip_submitted.py`: byte-identical copy of `submitted files/gossip.py` from the coursework folder.
- `results/historical_40step/`: byte-identical `summary.csv` and 24 per-configuration CSV files from the neighboring original `runs/` folder.
- `results/historical_200step/transcript.txt`: the commented historical terminal output extracted from the original `sanity_check.py`. Comment prefixes were removed. The surrounding single-process sanity-check program does not produce this distributed transcript; the comment is treated as an archived record, not newly executed output.
- `results/historical_200step/summary_from_transcript.csv`: parsed from the 24 `FINAL` lines in that transcript. Their recorded rounding is retained. World size and processed samples are derived from the run headers and 200-step / 64-batch settings visible in the transcript.
- The three-page jointly authored report, dated February 26, 2026, was read to check the table and experimental interpretation. Its accuracy values agree with the longer transcript to the displayed precision. The public Markdown summary replaces redistribution of the full homework report and prompt.

The neighboring 40-step CSV files use a different schema from the submitted program and contain different outcomes. Their exact generating revision and complete invocation were not preserved. The files are evidence of historical runs, not a claim that the current script will reproduce them byte for byte.

[`source_manifest.json`](source_manifest.json) records relative source labels and SHA-256 hashes. It deliberately contains no machine-specific home paths or personal contact information. Copied CSVs and the original source can be checked with `python verify_artifacts.py`.

## Changes to the runnable `gossip.py`

The core model, gradient update, graph construction, mixing, evaluation and disagreement logic are retained. The archival copy makes these changes reviewable:

1. Added selectors for a single topology, mode and communication budget, so the first command need not launch the whole sweep.
2. Added a deterministic synthetic dataset option and configurable hidden width for a small no-download execution check. Historical defaults remain unchanged. Torchvision is imported only when MNIST is requested.
3. Added explicit CPU-thread control (one thread per worker by default) to avoid oversubscribing a small machine. This changes execution configuration, not the SGD rule.
4. Kept summaries and plots inside the requested run directory, away from the archived results.
5. Added basic argument validation and `--no-plots`.
6. Clarified the `async` and `sync` docstrings without renaming the historical argument values or altering their mixing rules.

No historical results were regenerated. No mixing correction, sampler redesign, held-out evaluation, fresh benchmark, or performance optimization is claimed. For example, the original iterator / `sampler.set_epoch(step)` behavior is retained; changing it would create another experiment version.

## Collaboration

Both original authors remain credited. No reliable per-file contribution split was found in the available materials, so this package does not claim sole authorship for either person.

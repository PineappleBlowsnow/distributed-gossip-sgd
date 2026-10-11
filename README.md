# Training Decentralized Models with Gossip SGD

**MVA course — Training and Deploying Large-Scale Models**

**PyTorch Distributed / Gloo · decentralized optimization coursework · Ying JIN and Felipe VICENTIN**

How much communication is useful when several workers train local copies of a model? This project compares synchronous neighbor averaging and random pairwise gossip for an MNIST MLP, across **ring, grid, star and complete graphs** and three communication budgets.

The implementation connects graph-based consensus to a concrete distributed training loop: partitioned data, identical initialization, parameter exchanges, local gradient steps, and measurements of model disagreement and training-subset accuracy.

## What is included

- A multi-process CPU implementation using PyTorch Distributed's **Gloo** backend.
- Four graph topologies; synchronous averaging and pairwise edge activations; communication budgets `nu = 1, 2, 4`.
- Two separately preserved historical experiment sets, each containing **24 configurations**.
- A readable [experiment summary](docs/experiments.md), the [unchanged submitted source](archive/gossip_submitted.py), and [provenance / packaging notes](docs/provenance.md).
- A no-download distributed smoke-test entry point and a dependency-free checker for the archived results.

This is a joint MVA course project from *Training and Deploying Large-Scale Models*, not a production distributed-training library. The report credits both authors; the available records do not establish a per-file division of their work.

## Quick start

Python 3.10+ is recommended. Use Linux or WSL with a PyTorch build supporting Gloo. Install a matching `torch` / `torchvision` pair for your platform, then the optional plotting dependency:

```bash
python -m pip install -r requirements.txt
```

**Check the preserved results without installing PyTorch or downloading data:**

```bash
python verify_artifacts.py
```

**Check actual four-process communication without downloading data:**

```bash
python smoke_test.py
```

**Run a tiny synthetic training check:**

```bash
python gossip.py --synthetic --topology ring --mode sync --nu 1 --world 4 --world_grid 4 --steps 2 --batch 8 --hidden 16 --eval 1 --no-plots --run_dir runs/smoke
```

The synthetic option was added during portfolio preparation; it is not a historical experiment or a replacement for the MNIST evaluation. The distributed smoke and training check require PyTorch. They were **not executed during packaging**, because the available runtime did not contain PyTorch; see [validation](docs/validation.md).

**Run one MNIST configuration** (downloads the dataset on first use):

```bash
python gossip.py --topology ring --mode sync --nu 1 --steps 200 --batch 64 --lr 0.1 --eval 50 --no-plots --run_dir runs/ring_sync
```

Omit `--no-plots` to produce curves. To run all 24 configurations with the historical default model (784 → 256 → 256 → 10), omit `--topology`, `--mode` and `--nu`. Grid uses nine workers by default; the other graphs use four. The full sweep is a substantially longer run than the smoke checks.

Fresh outputs go under `--run_dir`, including `summary.csv` and optional `plots/`. Use a new run directory to retain previous runs. Dataset files and generated outputs are ignored by Git.

## How communication is implemented

Each worker starts from the same weights, mixes model parameters, then takes one SGD step on a rank-specific batch.

- **`sync`:** a node averages its own parameter snapshot with all neighbor snapshots. Exchanges are blocking point-to-point operations. This is synchronous in its update semantics, not concurrent all-neighbor communication.
- **`async`:** this historical argument name means **centrally scheduled pairwise gossip**. Rank 0 samples an edge and broadcasts it to all workers; the two endpoints average their parameters. There are `nu × number_of_edges` activations per SGD step. It is **not a fully asynchronous system** with independent clocks or stale gradients.
- **Disagreement energy:** the sum, over workers, of squared distances to the current arithmetic mean of the parameter vectors.

The synchronous rule is row-stochastic but is generally **not doubly stochastic on irregular graphs**, including the star and the 3×3 grid. It does not preserve the simple arithmetic mean in general. The Laplacian consensus analysis in the coursework should not be treated as an exact proof of this implementation's behavior.

## Historical results at a glance

| Preserved set | Steps per worker | Configurations | Final recorded accuracy |
|---|---:|---:|---:|
| Local CSV runs | 40 | 24 | 62.84–69.92% |
| Report-associated terminal transcript | 200 | 24 | 85.45–86.89% |

**These are different runs and must not be merged.** Accuracy is evaluated on a fixed **subset of the MNIST training split**, not the official test set. Distributed sampling pads the 2,048-example evaluation subset for the nine-worker grid, adding duplicate evaluations. The metric averages each worker's local model on its assigned evaluation examples; it is not the accuracy of one globally averaged model.

The longer run suggests that additional communication often lowers disagreement with only small accuracy changes in these small experiments. It does not establish a universal topology ranking or a wall-clock speedup: graph size differs, communication broadcasts have a cost, and no multi-seed uncertainty estimates were recorded. See [the full summary](docs/experiments.md).

## Attribution and reuse

Coursework authors: **Ying JIN and Felipe VICENTIN**. Portfolio documentation and execution conveniences were added in September 2026; the original submission is preserved separately. No dataset, instructor prompt, course slides, credentials, or Python environment is included.

No license was present in the source materials, and this repository does not add an open-source license or grant new reuse rights. See [NOTICE.md](NOTICE.md).


[Back to the MVA project portfolio](https://github.com/PineappleBlowsnow/mva-portfolio)

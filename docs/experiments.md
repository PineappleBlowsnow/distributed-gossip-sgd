# Experiment summary

## Question and setup

The coursework combines graph-Laplacian consensus analysis with a small distributed SGD experiment. Each worker trains a three-layer ReLU MLP (784 → 256 → 256 → 10) on a partition of the MNIST training set using CPU PyTorch Distributed / Gloo. The recorded longer-run setup uses batch size 64, learning rate 0.1 and 200 local updates. Ring, star and complete graphs use four workers; the grid uses nine.

At each iteration, communication occurs before a local gradient update. `nu` increases the amount of mixing: synchronous mode applies `nu` rounds of neighbor averaging, while pairwise mode performs `nu × |E|` sampled edge activations, with an all-rank broadcast selecting every edge.

The theoretical part studies Laplacian positive semidefiniteness, spectral gaps, disagreement contraction and edge-activation budgets. It is useful motivation. It is not an implementation-correctness proof: the synchronous code uses degree-normalized neighbor averaging rather than a common symmetric `I - alpha L` matrix, and the pairwise code is centrally scheduled rather than an independently timed network.

## Recorded 200-step outcomes

Each cell below lists final accuracy for `nu = 1 / 2 / 4`, respectively. Values are transcribed from the historical terminal output, not rerun in September 2026.

| Graph | Workers | Sync accuracy | Pairwise (`async`) accuracy |
|---|---:|---|---|
| Ring | 4 | 0.8564 / 0.8564 / 0.8550 | 0.8613 / 0.8550 / 0.8555 |
| Grid | 9 | 0.8660 / 0.8650 / 0.8650 | 0.8689 / 0.8670 / 0.8626 |
| Star | 4 | 0.8604 / 0.8604 / 0.8604 | 0.8623 / 0.8589 / 0.8564 |
| Complete | 4 | 0.8555 / 0.8555 / 0.8555 | 0.8594 / 0.8545 / 0.8555 |

Across the four-worker graphs, final disagreement energy is approximately 0.039–0.046. The grid has about 0.107–0.114. Greater `nu` generally reduces grid disagreement, while accuracy changes remain small and are not uniformly positive. Complete-graph synchronous averaging already reaches consensus in one mixing round, so repeated rounds before the next gradient step give very similar results.

These observations are narrow: the grid has a different worker count and processes more samples per iteration, and disagreement is a sum rather than a per-worker average. The numbers do not isolate topology from scale. There are no multiple seeds, confidence intervals, held-out test results or controlled wall-clock comparisons.

## Separate 40-step archive

The adjacent CSV archive contains 24 shorter runs with checkpoints at steps 0, 10, 20, 30 and 39. It records final training-subset accuracy between 0.62841796875 and 0.69921875, and has different initial values from the longer transcript. The files have been preserved under [`historical_40step`](../results/historical_40step/) rather than appended to the 200-step logs.

Their column labels are `samples_processed`, `disagreement_energy` and `train_acc_eval`, whereas the submitted script writes `samples`, `E` and `acc`. Exact code/version provenance for that shorter set is unavailable. Its summary includes `reach_samples`, but the underlying threshold definition is not recorded; do not interpret that field as a speed-to-target result.

## What accuracy means here

The script samples 2,048 examples from **the training split** for evaluation. Each rank evaluates its own model on part of that subset, then correct / total counts are summed. This is neither a held-out test metric nor the metric of the mean model. For nine ranks, `DistributedSampler` pads the evaluation subset to an evenly divisible size, duplicating four examples.

Disagreement is `sum_i ||theta_i - mean(theta)||^2`, measured after the local gradient step. A low value indicates similar parameter vectors; it does not, by itself, establish better generalization.

## Limits worth investigating next

- Use a symmetric doubly stochastic mixing matrix if mean-preserving synchronous consensus is intended.
- Hold graph size and processed samples fixed when comparing topology.
- Evaluate one defined model on held-out MNIST test data without sampler padding.
- Measure actual bytes/messages, synchronization and wall-clock time, including edge-selection broadcasts.
- Repeat seeds and compare uncertainty before drawing an accuracy ranking.

These are future experiments, not completed features or results.

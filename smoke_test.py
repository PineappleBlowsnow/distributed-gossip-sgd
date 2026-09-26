"""Four-process CPU/Gloo checks of the actual mixing functions; no data download.

Added for portfolio preparation, not part of the historical coursework results.
Run with a PyTorch installation: python smoke_test.py
"""
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gossip import async_gossip, build_graph, disagreement, sync_gossip


def check_worker(rank, world, store_path):
    torch.set_num_threads(1)
    store = dist.FileStore(store_path, world)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    try:
        values = torch.arange(world * 3, dtype=torch.float32).reshape(world, 3)
        for topology in ("ring", "grid", "star", "complete"):
            edges, neighbors = build_graph(topology, world)
            x = values[rank].clone()
            expected = values[[rank] + neighbors[rank]].mean(dim=0)
            actual = sync_gossip(x, neighbors, rank)
            torch.testing.assert_close(actual, expected)
            # Every rank participates in the same broadcast, as in the trainer.
            edge = torch.tensor(edges[-1], dtype=torch.long) if rank == 0 else torch.empty(2, dtype=torch.long)
            dist.broadcast(edge, 0)
            i, j = edge.tolist()
            actual = async_gossip(x, (i, j), rank)
            expected = (values[i] + values[j]) / 2 if rank in (i, j) else x
            torch.testing.assert_close(actual, expected)
            # Pairwise mixing must preserve the vector sum across all workers.
            total = actual.clone()
            dist.all_reduce(total)
            torch.testing.assert_close(total, values.sum(dim=0))
            expected_energy = ((values - values.mean(dim=0)) ** 2).sum().item()
            assert abs(disagreement(x) - expected_energy) < 1e-5
            dist.barrier()
        if rank == 0:
            print("PASS: four topologies, synchronous snapshots, pairwise updates, mean preservation and disagreement")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if not dist.is_available() or not dist.is_gloo_available():
        raise SystemExit("This smoke test requires a PyTorch build with the Gloo backend.")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(check_worker, args=(4, str(Path(directory) / "store")), nprocs=4, join=True)

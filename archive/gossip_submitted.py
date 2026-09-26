#!/usr/bin/env python3
"""
gossip.py - Gossip-based distributed SGD using gloo backend 

Implements Q4: synchronous and asynchronous gossip for distributed training of a
3-layer MLP on MNIST. Compares ring, grid, star, complete topologies with nu in {1,2,4}.

Usage (Linux/WSL):
  python gossip.py
  python gossip.py --steps 200 --batch 64 --lr 0.1

Output: plots in plots, intermediate outputs, summary printed at the end
"""
import os, math, csv, argparse, random, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

# ---------------- Graphs ----------------
def edges_ring(n): return [(i, (i + 1) % n) for i in range(n)]
def edges_star(n): return [(0, i) for i in range(1, n)]
def edges_complete(n): return [(i, j) for i in range(n) for j in range(i + 1, n)]

def edges_grid(n):
    s = int(math.isqrt(n))
    if s * s != n: raise ValueError("grid requires perfect square")
    idx = lambda r, c: r * s + c
    E = []
    for r in range(s):
        for c in range(s):
            if r + 1 < s: E.append((idx(r, c), idx(r + 1, c)))
            if c + 1 < s: E.append((idx(r, c), idx(r, c + 1)))
    return E

def build_graph(topo, n):
    if topo == "ring": E = edges_ring(n)
    elif topo == "grid": E = edges_grid(n)
    elif topo == "star": E = edges_star(n)
    elif topo == "complete": E = edges_complete(n)
    else: raise ValueError("unknown topology")
    neigh = [[] for _ in range(n)]
    for i, j in E: neigh[i].append(j); neigh[j].append(i)
    return E, neigh

# ---------------- Model ----------------
class MLP(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(784, h), nn.Linear(h, h), nn.Linear(h, 10)
    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))))

def flatten(m): return torch.cat([p.data.view(-1) for p in m.parameters()])
def unflatten(m, f):
    o = 0
    for p in m.parameters(): p.data.copy_(f[o:o+p.numel()].view_as(p)); o += p.numel()

# ---------------- Gossip (gloo) ----------------
def send_recv(t, peer, rank):
    r = torch.empty_like(t)
    if rank < peer: dist.send(t, peer); dist.recv(r, peer)
    else: dist.recv(r, peer); dist.send(t, peer)
    return r

def sync_gossip(x, neigh, rank):
    """Synchronous: average with all neighbors simultaneously."""
    s = x.clone()
    for j in neigh[rank]: s += send_recv(x, j, rank)
    return s / (len(neigh[rank]) + 1)

def async_gossip(x, edge, rank):
    """Asynchronous: pairwise averaging on activated edge."""
    i, j = edge
    if rank != i and rank != j: return x
    peer = j if rank == i else i
    return 0.5 * (x + send_recv(x, peer, rank))

# ---------------- Metrics ----------------
@torch.no_grad()
def disagreement(x):
    w = dist.get_world_size()
    avg = x.clone(); dist.all_reduce(avg); avg /= w
    loc = ((x - avg)**2).sum(); dist.all_reduce(loc)
    return loc.item()

@torch.no_grad()
def accuracy(model, loader):
    c, t = torch.tensor(0), torch.tensor(0)
    for xb, yb in loader:
        c += (model(xb).argmax(1) == yb).sum(); t += yb.size(0)
    dist.all_reduce(c); dist.all_reduce(t)
    return (c.float() / t.float()).item()

# ---------------- Worker ----------------
def worker(rank, cfg, store_path):
    world = cfg["world"]
    store = dist.FileStore(store_path, world)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    
    topo, mode, nu = cfg["topo"], cfg["mode"], cfg["nu"]
    E, neigh = build_graph(topo, world); m = len(E)
    
    ds = datasets.MNIST(cfg["data"], train=True, download=False, transform=transforms.ToTensor())
    
    # Use rank-specific seed for data sampling (different data per worker)
    torch.manual_seed(cfg["seed"] + rank); random.seed(cfg["seed"] + rank)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, seed=cfg["seed"])
    loader = DataLoader(ds, cfg["batch"], sampler=sampler, drop_last=True, num_workers=0)
    it = iter(loader)
    
    g = torch.Generator().manual_seed(cfg["seed"])
    eval_sub = Subset(ds, torch.randperm(len(ds), generator=g)[:2048].tolist())
    eval_samp = DistributedSampler(eval_sub, world, rank, shuffle=False)
    eval_ld = DataLoader(eval_sub, cfg["batch"], sampler=eval_samp, num_workers=0)
    
    # CRITICAL: All workers must start with SAME initial weights
    torch.manual_seed(cfg["seed"])  # Same seed for model init across all ranks
    model = MLP(); model.train()
    
    # Restore rank-specific randomness for async edge selection
    random.seed(cfg["seed"] + rank)
    
    if rank == 0:
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        with open(cfg["out"], "w") as f: csv.writer(f).writerow(["step","samples","E","acc"])
        print(f"RUN {topo} {mode} nu={nu} n={world} |E|={m}")
    
    for step in range(cfg["steps"]):
        sampler.set_epoch(step)
        try: xb, yb = next(it)
        except StopIteration: it = iter(loader); xb, yb = next(it)
        
        x = flatten(model)
        # Gossip communication (nu controls budget)
        if mode == "sync":
            for _ in range(nu): x = sync_gossip(x, neigh, rank)
        else:  # async: nu*|E| pairwise activations
            for _ in range(nu * m):
                if rank == 0: e = torch.tensor(E[random.randrange(m)], dtype=torch.long)
                else: e = torch.empty(2, dtype=torch.long)
                dist.broadcast(e, 0)
                x = async_gossip(x, (e[0].item(), e[1].item()), rank)
        unflatten(model, x)
        
        # Local SGD step
        model.zero_grad()
        F.cross_entropy(model(xb), yb).backward()
        with torch.no_grad():
            for p in model.parameters(): p -= cfg["lr"] * p.grad
        
        # Evaluate
        if step % cfg["eval"] == 0 or step == cfg["steps"] - 1:
            E_val = disagreement(flatten(model))
            model.eval(); acc = accuracy(model, eval_ld); model.train()
            samples = (step + 1) * world * cfg["batch"]
            if rank == 0:
                with open(cfg["out"], "a") as f: csv.writer(f).writerow([step, samples, E_val, acc])
                print(f"  step={step} samples={samples} E={E_val:.4e} acc={acc:.4f}")
    
    dist.barrier()
    if rank == 0:
        with open(cfg["out"]) as f: results = list(csv.DictReader(f))
        final_E, final_acc = float(results[-1]["E"]), float(results[-1]["acc"])
        with open(cfg["summary"], "a") as f:
            csv.writer(f).writerow([topo, mode, nu, world, m, final_E, final_acc])
        print(f"FINAL {topo} {mode} nu={nu} E={final_E:.4e} acc={final_acc:.4f}")
    dist.destroy_process_group()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=int, default=20)
    ap.add_argument("--data", type=str, default="./data")
    ap.add_argument("--run_dir", type=str, default="runs")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--world_grid", type=int, default=9)
    args = ap.parse_args()

    topos, modes, nus = ["ring", "grid", "star", "complete"], ["sync", "async"], [1, 2, 4]
    os.makedirs(args.run_dir, exist_ok=True)
    summary = "summary.csv"
    with open(summary, "w") as f: csv.writer(f).writerow(["topo","mode","nu","n","m","E","acc"])

    # Pre-download MNIST to avoid race condition
    print("Downloading MNIST dataset...")
    datasets.MNIST(args.data, train=True, download=True)
    print("Dataset ready.\n")

    for topo in topos:
        world = args.world_grid if topo == "grid" else args.world
        for mode in modes:
            for nu in nus:
                cfg = {"topo": topo, "mode": mode, "nu": nu, "world": world,
                       "steps": args.steps, "batch": args.batch, "lr": args.lr,
                       "seed": args.seed, "eval": args.eval, "data": args.data,
                       "out": f"{args.run_dir}/{topo}_{mode}_nu{nu}.csv", "summary": summary}
                with tempfile.TemporaryDirectory() as td:
                    mp.spawn(worker, args=(cfg, f"{td}/store"), nprocs=world, join=True)

    # Print summary table
    print("\n" + "="*70)
    print("KEY QUANTITIES SUMMARY")
    print("="*70)
    print(f"{'Topology':<10} {'Mode':<6} {'nu':<3} {'n':<3} {'|E|':<4} {'Final E':<12} {'Accuracy':<10}")
    print("-"*70)
    with open(summary) as f:
        for row in csv.DictReader(f):
            print(f"{row['topo']:<10} {row['mode']:<6} {row['nu']:<3} {row['n']:<3} "
                  f"{row['m']:<4} {float(row['E']):<12.4e} {float(row['acc']):<10.4f}")
    print("="*70)

    # Generate plots
    try:
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)
        for topo in topos:
            files = [f for f in os.listdir(args.run_dir) if f.startswith(f"{topo}_")]
            for metric, col, ylabel in [("E", "E", "Disagreement Energy"), ("acc", "acc", "Accuracy")]:
                plt.figure(figsize=(8, 5))
                for fname in sorted(files):
                    with open(f"{args.run_dir}/{fname}") as f:
                        rows = list(csv.DictReader(f))
                    x, y = [int(r["samples"]) for r in rows], [float(r[col]) for r in rows]
                    label = fname.replace(f"{topo}_", "").replace(".csv", "")
                    plt.plot(x, y, 'o-', markersize=3, label=label)
                plt.xlabel("Samples Processed"); plt.ylabel(ylabel)
                plt.title(f"{topo.capitalize()}: {ylabel}"); plt.legend(); plt.grid(alpha=0.3)
                plt.tight_layout(); plt.savefig(f"plots/{topo}_{metric}.png", dpi=150); plt.close()
        print(f"\nPlots saved: plots/{{ring,grid,star,complete}}_{{E,acc}}.png")
    except ImportError: print("matplotlib not available, skipping plots")
    print("DONE.")

if __name__ == "__main__":
    main()



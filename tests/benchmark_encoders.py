import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from marlprp.models.policy_args import TransformerParams
from marlprp.models.encoder.matnet import MatNetEncoderLayer


torch.manual_seed(0)
device = "cuda:0"


def benchmark_train_step(layer, shelf, prod, supply, n_warmup=3, n_runs=10):
    # Make sure inputs require grad if needed (only matters if layer depends on them)
    if shelf.requires_grad is False:
        shelf = shelf.clone().requires_grad_()
    if prod.requires_grad is False:
        prod = prod.clone().requires_grad_()

    torch.cuda.synchronize()

    # Warm-up
    for _ in range(n_warmup):
        out = layer(shelf, prod, cost_mat=supply)
        loss = out[0].sum()
        loss.backward()
        layer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed section
    t0 = time.time()
    for _ in range(n_runs):
        out = layer(shelf, prod, cost_mat=supply)
        loss = out[0].sum()
        loss.backward()
        layer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**2
    return (t1 - t0) / n_runs * 1000, mem  # ms, MB


# --- Benchmark Function ---
@torch.no_grad()
def benchmark(layer, shelf, prod, supply, n_warmup=3, n_runs=10):
    torch.cuda.synchronize()
    for _ in range(n_warmup): layer(shelf, prod, cost_mat=supply)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_runs): layer(shelf, prod, cost_mat=supply)
    torch.cuda.synchronize()
    t1 = time.time()
    mem = torch.cuda.max_memory_allocated() / 1024**2
    return (t1 - t0) / n_runs * 1000, mem  # ms, MB



# --- Setup ---
bench_fn = benchmark
B, S, P, D, H = 128, 50, 1000, 256, 8
num_pairs = 1000
density = num_pairs/(S*P)

shelf = torch.randn(B, S, D, device=device).float()
prod  = torch.randn(B, P, D, device=device).float()
supply  = (torch.rand(B, S, P, device=device) < density).float()

kwargs = {
    'env': None,
    'embed_dim': D,
    'num_heads': H,
    'chunk_ms_scores_batch': 0,
    'bias': True,
    'param_sharing': False,
}


dense_encoder = MatNetEncoderLayer(TransformerParams(ms_sparse_attn=False, **kwargs)).to(device)
sparse_encoder = MatNetEncoderLayer(TransformerParams(ms_sparse_attn=True, use_self_attn=False, **kwargs)).to(device)

# --- Benchmark ---
try:
    dense_time, dense_mem = bench_fn(dense_encoder, shelf, prod, supply)
except:
    dense_time = 1000000000
    dense_mem = 100000000

sparse_time, sparse_mem = bench_fn(sparse_encoder, shelf, prod, supply)


print("\n---- Dense vs Sparse Attention ----")

print(f"Dense attention:  {dense_time:.2f} ms / {dense_mem:.1f} MB")
print(f"Sparse attention: {sparse_time:.2f} ms / {sparse_mem:.1f} MB")
print(f"Sparsity: {density*100:.1f}%  |  Speedup: {dense_time/sparse_time:.2f}x  |  Mem ratio: {dense_mem/sparse_mem:.2f}x")


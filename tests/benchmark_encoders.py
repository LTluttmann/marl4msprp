import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from marlprp.models.policy_args import TransformerParams
from marlprp.models.encoder.matnet import MatNetEncoderLayer


torch.manual_seed(0)
device = "cuda:0"




# --- Benchmark Function ---
@torch.no_grad()
def benchmark(layer, shelf, prod, supply, n_warmup=10, n_runs=50):
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
B, S, P, D, H = 25000, 25, 18, 256, 8
num_pairs = 50
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
    'param_sharing': True,
}


dense_encoder = MatNetEncoderLayer(TransformerParams(ms_sparse_attn=False, **kwargs)).to(device)
sparse_encoder = MatNetEncoderLayer(TransformerParams(ms_sparse_attn=True, **kwargs)).to(device)

# --- Benchmark ---
dense_time, dense_mem = benchmark(dense_encoder, shelf, prod, supply)
sparse_time, sparse_mem = benchmark(sparse_encoder, shelf, prod, supply)


print("\n---- Dense vs Sparse Attention ----")

print(f"Dense attention:  {dense_time:.2f} ms / {dense_mem:.1f} MB")
print(f"Sparse attention: {sparse_time:.2f} ms / {sparse_mem:.1f} MB")
print(f"Sparsity: {density*100:.1f}%  |  Speedup: {dense_time/sparse_time:.2f}x  |  Mem ratio: {dense_mem/sparse_mem:.2f}x")


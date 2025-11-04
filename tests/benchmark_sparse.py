import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from marlprp.models.policy_args import TransformerParams
from marlprp.models.encoder.cross_attention import EfficientMixedScoreMultiHeadAttentionLayer, MixedScoreMultiHeadAttention
from marlprp.models.encoder.sparse import SparseCrossAttention, EfficientSparseCrossAttention

torch.manual_seed(0)
device = "cuda:0"




# --- Benchmark Function ---
def benchmark(layer, shelf, prod, supply, n_warmup=10, n_runs=50):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    for _ in range(n_warmup): layer(shelf, prod, cost_mat=supply)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_runs): layer(shelf, prod, cost_mat=supply)
    torch.cuda.synchronize()
    t1 = time.time()
    mem = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.empty_cache()
    return (t1 - t0) / n_runs * 1000, mem  # ms, MB

# --- Setup ---
B, S, P, D, H = 5000, 25, 18, 256, 8
num_pairs = 50
density = num_pairs/(S*P)

shelf = torch.randn(B, S, D, device=device).float()
prod  = torch.randn(B, P, D, device=device).float()
supply  = (torch.rand(B, S, P, device=device) < density).float()


params = TransformerParams(
    env=None,
    embed_dim=D,
    num_heads=H,
    chunk_ms_scores_batch=0,
    bias=True,
)


# --- Setup Layers ---
dense_layer = MixedScoreMultiHeadAttention(params).to(device)
sparse_layer = SparseCrossAttention(params).to(device)
# --- Benchmark ---
dense_time, dense_mem = benchmark(dense_layer, shelf, prod, supply)
sparse_time, sparse_mem = benchmark(sparse_layer, shelf, prod, supply)


print("\n---- Dense vs Sparse Attention ----")

print(f"Dense attention:  {dense_time:.2f} ms / {dense_mem:.1f} MB")
print(f"Sparse attention: {sparse_time:.2f} ms / {sparse_mem:.1f} MB")
print(f"Sparsity: {density*100:.1f}%  |  Speedup: {dense_time/sparse_time:.2f}x  |  Mem ratio: {dense_mem/sparse_mem:.2f}x")

# --- Setup Efficient Layers ---
efficient_dense_layer  = EfficientMixedScoreMultiHeadAttentionLayer(params).to(device)
efficient_sparse_layer = EfficientSparseCrossAttention(params).to(device)
# --- Benchmark ---
eff_dense_time, eff_dense_mem = benchmark(efficient_dense_layer, shelf, prod, supply)
eff_sparse_time, eff_sparse_mem = benchmark(efficient_sparse_layer, shelf, prod, supply)
print("\n---- Efficient Dense vs. Sparse Attention ----")

print(f"Efficient Dense attention:  {eff_dense_time:.2f} ms / {eff_dense_mem:.1f} MB")
print(f"Efficient Sparse attention: {eff_sparse_time:.2f} ms / {eff_sparse_mem:.1f} MB")
print(f"Sparsity: {density*100:.1f}%  |  Speedup: {eff_dense_time/eff_sparse_time:.2f}x  |  Mem ratio: {eff_dense_mem/eff_sparse_mem:.2f}x")
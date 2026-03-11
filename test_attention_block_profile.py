"""
Usage:  python test_attention_block_profile.py <experiment>
  experiment: 1_baseline | 2_vectorized | 3_compile_default | 4_compile_fullgraph | 5_triton

Each run is wrapped by nsys externally via run_all_profiles.sh
"""

import sys
import torch
import torch.cuda.amp as amp
import nvtx
import wan.modules.model as M
from wan.modules.model import WanAttentionBlock, rope_params
from wan.modules.rope_triton import rope_apply_triton

B, L, D, FFN_D, HEADS = 1, 4096, 2048, 8192, 16
GRID = (4, 32, 32)
DTYPE = torch.bfloat16
WARMUP, REPEATS = 5, 20

block = WanAttentionBlock("t2v_cross_attn", D, FFN_D, HEADS).cuda().to(DTYPE).eval()

d = D // HEADS
inputs = dict(
    x=torch.randn(B, L, D, device="cuda", dtype=DTYPE),
    e=torch.randn(B, 6, D, device="cuda", dtype=torch.float32),
    seq_lens=torch.tensor([L] * B, dtype=torch.long, device="cuda"),
    grid_sizes=torch.tensor([GRID] * B, dtype=torch.long, device="cuda"),
    freqs=torch.cat([rope_params(1024, d - 4*(d//6)),
                     rope_params(1024, 2*(d//6)),
                     rope_params(1024, 2*(d//6))], dim=1).cuda(),
    context=torch.randn(B, 512, D, device="cuda", dtype=DTYPE),
    context_lens=None,
)


@amp.autocast(enabled=False)
def rope_apply_old(x, f, h, w, freqs):
    """Original loop-based rope_apply, adapted to new (x, f, h, w, freqs) signature."""
    n, c = x.size(2), x.size(3) // 2
    seq_len = f * h * w
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i in range(x.size(0)):
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


rope_vectorized = M.rope_apply


def run(tag):
    for _ in range(WARMUP):
        with torch.no_grad(), amp.autocast(dtype=DTYPE):
            block(**inputs)
    torch.cuda.synchronize()

    for i in range(REPEATS):
        with torch.no_grad(), amp.autocast(dtype=DTYPE), nvtx.annotate(f"{tag}/iter_{i}"):
            block(**inputs)
    torch.cuda.synchronize()


exp = sys.argv[1]

if exp == "1_baseline":
    M.rope_apply = rope_apply_old
    run("baseline")

elif exp == "2_vectorized":
    M.rope_apply = rope_vectorized
    run("vectorized")

elif exp == "3_compile_default":
    M.rope_apply = torch.compile(rope_vectorized, mode="default")
    run("compile_default")

elif exp == "4_compile_fullgraph":
    M.rope_apply = torch.compile(rope_vectorized, mode="default", fullgraph=True)
    run("compile_fullgraph")

elif exp == "5_triton":
    M.rope_apply = rope_apply_triton
    run("triton")

else:
    print(f"Unknown experiment: {exp}")
    sys.exit(1)

"""
nsys profile -t cuda,nvtx --force-overwrite true \
    -o attention_block python test_attention_block_profile.py
"""

import torch
import torch.cuda.amp as amp
import torch.cuda.nvtx as nvtx
import wan.modules.model as M
from wan.modules.model import WanAttentionBlock, rope_params

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

rope_eager = M.rope_apply
rope_compiled = torch.compile(M.rope_apply, mode="default")


def set_rope(fn):
    M.rope_apply = fn


def run(tag):
    for _ in range(WARMUP):
        with torch.no_grad(), amp.autocast(dtype=DTYPE):
            block(**inputs)
    torch.cuda.synchronize()

    for i in range(REPEATS):
        with torch.no_grad(), amp.autocast(dtype=DTYPE), nvtx.annotate(f"{tag}/iter_{i}"):
            block(**inputs)
    torch.cuda.synchronize()


set_rope(rope_eager)
run("baseline")

set_rope(rope_compiled)
run("compiled_rope")

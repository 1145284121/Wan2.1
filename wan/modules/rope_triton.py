import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(
    X_ptr, OUT_ptr, COS_ptr, SIN_ptr,
    seq_len, L, n_heads, half_dim,
    stride_xb, stride_xl, stride_xn, stride_xd,
    stride_cb, stride_cc,
    BLOCK_C: tl.constexpr,
):
    pos = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // n_heads
    n_idx = pid_bh % n_heads

    x_base = b_idx * stride_xb + pos * stride_xl + n_idx * stride_xn

    cols = tl.arange(0, BLOCK_C)
    mask = cols < half_dim * 2

    if pos >= seq_len:
        x_val = tl.load(X_ptr + x_base + cols * stride_xd, mask=mask)
        tl.store(OUT_ptr + x_base + cols * stride_xd, x_val, mask=mask)
        return

    pair_idx = cols // 2
    is_odd = cols % 2

    x_even = tl.load(X_ptr + x_base + (pair_idx * 2) * stride_xd, mask=mask)
    x_odd = tl.load(X_ptr + x_base + (pair_idx * 2 + 1) * stride_xd, mask=mask)

    cos_val = tl.load(COS_ptr + pos * stride_cb + pair_idx * stride_cc, mask=mask)
    sin_val = tl.load(SIN_ptr + pos * stride_cb + pair_idx * stride_cc, mask=mask)

    rot_even = x_even * cos_val - x_odd * sin_val
    rot_odd = x_even * sin_val + x_odd * cos_val

    out_val = tl.where(is_odd != 0, rot_odd, rot_even)
    tl.store(OUT_ptr + x_base + cols * stride_xd, out_val, mask=mask)


_cos_sin_cache = {}


def _get_cos_sin(f, h, w, freqs, half_dim):
    key = (f, h, w, freqs.data_ptr())
    if key in _cos_sin_cache:
        return _cos_sin_cache[key]

    c_f = half_dim - 2 * (half_dim // 3)
    c_h = half_dim // 3

    freqs_real = torch.view_as_real(freqs).float()
    cos_all = freqs_real[:, :, 0]
    sin_all = freqs_real[:, :, 1]

    cos_f = cos_all[:f, :c_f].reshape(f, 1, 1, c_f).expand(f, h, w, c_f)
    sin_f = sin_all[:f, :c_f].reshape(f, 1, 1, c_f).expand(f, h, w, c_f)
    cos_h = cos_all[:h, c_f:c_f+c_h].reshape(1, h, 1, c_h).expand(f, h, w, c_h)
    sin_h = sin_all[:h, c_f:c_f+c_h].reshape(1, h, 1, c_h).expand(f, h, w, c_h)
    cos_w = cos_all[:w, c_f+c_h:].reshape(1, 1, w, -1).expand(f, h, w, -1)
    sin_w = sin_all[:w, c_f+c_h:].reshape(1, 1, w, -1).expand(f, h, w, -1)

    seq_len = f * h * w
    cos_table = torch.cat([cos_f, cos_h, cos_w], dim=-1).reshape(seq_len, half_dim).contiguous()
    sin_table = torch.cat([sin_f, sin_h, sin_w], dim=-1).reshape(seq_len, half_dim).contiguous()

    _cos_sin_cache[key] = (cos_table, sin_table)
    return cos_table, sin_table


def rope_apply_triton(x, f, h, w, freqs):
    B, L, n_heads, head_dim = x.shape
    half_dim = head_dim // 2
    seq_len = f * h * w

    cos_table, sin_table = _get_cos_sin(f, h, w, freqs, half_dim)

    out = torch.empty_like(x, dtype=torch.float32)
    x_f32 = x.float()

    BLOCK_C = triton.next_power_of_2(head_dim)
    grid = (L, B * n_heads)

    _rope_fwd_kernel[grid](
        x_f32, out, cos_table, sin_table,
        seq_len, L, n_heads, half_dim,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        cos_table.stride(0), cos_table.stride(1),
        BLOCK_C=BLOCK_C,
    )
    return out

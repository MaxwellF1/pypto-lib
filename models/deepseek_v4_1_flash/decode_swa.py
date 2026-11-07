# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SWA decode attention with mHC."""


import pypto.language as pl
from qkv_proj_rope import PROJ_M_TILE as QKV_PROJ_M_TILE
from hc import LINEAR_T_TILE as HC_LINEAR_T_TILE, LINEAR_OK as HC_LINEAR_OK, MIX_PAD as HC_MIX_PAD
from config import FLASH as M, ORI_BLOCKS_DYN, WINDOW_CACHE_GROUP
from sparse_attention import _attend_window_project
from qkv_proj_rope import quantize_projection_input_decode as quantize_projection_input, q_proj_rope_decode as q_proj_rope, kv_proj_decode as kv_proj, window_publish_decode as window_publish
from hc import hc_head_norm_decode as hc_head_norm, hc_project_decode as hc_project, hc_sinkhorn_decode as hc_sinkhorn, zero_bf16_padding_decode as zero_bf16_padding, hc_post_decode as hc_post


# model config
LOCAL_H = M.num_attention_heads
LOCAL_O_GROUPS = M.o_groups
LOCAL_O_WIDTH = M.o_groups * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
MX_GROUP = M.weight_block_size[0]

# tiling


from config import MOE_TOKENS as DECODE_GRAPH_CAPACITY


@pl.jit.inline(auto_scope=False)
def decode_swa(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    attention_output: pl.Tensor[[DECODE_GRAPH_CAPACITY, D], pl.BF16],
    attention_hidden: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    attention_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    x_norm: pl.Tensor,
    qr: pl.Tensor,
    q: pl.Tensor,
    kv_projection: pl.Tensor,
    qkv_x_norm_payload: pl.Tensor,
    qkv_x_norm_scales: pl.Tensor,
    qb_workspace_payload: pl.Tensor,
    qb_workspace_scales: pl.Tensor,
    window_scale_scratch: pl.Tensor,
    window_attended: pl.Tensor,
    hc_inv_rms: pl.Tensor,
    hc_partials: pl.Tensor,
    window_blocks: pl.Scalar[pl.INDEX],
):
    """Run one SWA attention sublayer and emit the staggered mix for the next sublayer."""
    x_ready = hc_head_norm(x_hc, incoming_pre_mix, attn_norm_weight, x_norm)

    zero_bf16_padding(x_norm, pl.min(num_tokens, pl.tensor.dim(x_norm, 0)))
    proj_payload, proj_scales = quantize_projection_input(x_norm, num_tokens, qkv_x_norm_payload, qkv_x_norm_scales)
    qr, q, qb_payload, qb_scales = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, q,
        num_tokens,
     qb_workspace_payload, qb_workspace_scales)
    kv_projection = kv_proj(proj_payload, proj_scales, wkv, wkv_scale, kv_projection, num_tokens)
    cache_ready = window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache, window_cache_scale,
        num_tokens, x_ready,
     window_scale_scratch, cache_blocks=window_blocks)
    _attend_window_project(
        q, window_indices, window_cache, window_cache_scale, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, attention_output,
        num_tokens, cache_ready,
     window_attended, physical_capacity=DECODE_GRAPH_CAPACITY)
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, hf32=True, inv_rms=hc_inv_rms, mixes_partials=hc_partials)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden, deps=hc_post_deps)
    return attention_hidden, attention_pre_mix


@pl.jit
def l2_decode_swa(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    attention_output: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Dispatch one production SWA composition to a device-selected JIT entry."""
    swa_post_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT], dtype=pl.FP32)
    swa_residual_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
    swa_x_norm = pl.create_tensor([DECODE_GRAPH_CAPACITY, D], dtype=pl.BF16)
    swa_qr = pl.create_tensor([DECODE_GRAPH_CAPACITY, Q_LORA], dtype=pl.BF16)
    swa_q = pl.create_tensor([DECODE_GRAPH_CAPACITY, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    swa_kv_projection = pl.create_tensor([DECODE_GRAPH_CAPACITY, HEAD_DIM], dtype=pl.BF16)
    qkv_rows = ((pl.tensor.dim(x_hc, 0) + QKV_PROJ_M_TILE - 1) // QKV_PROJ_M_TILE) * QKV_PROJ_M_TILE
    qkv_x_norm_payload = pl.create_tensor([qkv_rows, D], dtype=pl.FP8E4M3FN)
    qkv_x_norm_scales = pl.create_tensor([1, qkv_rows * (D // 32)], dtype=pl.FP8E8M0)
    qb_rows = ((pl.tensor.dim(x_hc, 0) + QKV_PROJ_M_TILE - 1) // QKV_PROJ_M_TILE) * QKV_PROJ_M_TILE
    qb_workspace_payload = pl.create_tensor([qb_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_workspace_scales = pl.create_tensor([1, qb_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    window_scale_scratch = pl.create_tensor([pl.tensor.dim(window_cache_scale, 0), 128, 1, HEAD_DIM // 32], dtype=pl.BF16)
    window_attended = pl.create_tensor([pl.tensor.dim(x_hc, 0), LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    hc_workspace_rows = ((pl.tensor.dim(x_hc, 0) + HC_LINEAR_T_TILE - 1) // HC_LINEAR_T_TILE) * HC_LINEAR_T_TILE
    hc_inv_rms = pl.create_tensor([hc_workspace_rows, 1], dtype=pl.FP32)
    hc_partials = pl.create_tensor([HC_LINEAR_OK * hc_workspace_rows, HC_MIX_PAD], dtype=pl.FP32)
    with pl.graph("lib_decode_swa"):
        decode_swa(
            x_hc, incoming_pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale,
            attention_output, attention_hidden, attention_pre_mix,
            num_tokens,
            swa_post_mix, swa_residual_mix, swa_x_norm, swa_qr, swa_q, swa_kv_projection,
            qkv_x_norm_payload, qkv_x_norm_scales,
         qb_workspace_payload, qb_workspace_scales, window_scale_scratch, window_attended, hc_inv_rms, hc_partials, window_blocks=pl.tensor.dim(window_cache_scale, 0))
    return attention_hidden, attention_pre_mix


@pl.jit.host
def l3_decode_swa(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    attention_output: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY, D], pl.BF16]],
    attention_hidden: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    attention_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    # MX_B_NN weight-scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    l2_decode_swa(
        x_hc, incoming_pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_r, q_norm_weight, wq_b, wq_b_scale_r,
        wkv, wkv_scale_r, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_r,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        attention_output, attention_hidden, attention_pre_mix,
        num_tokens, device=0,
    )


def _padded_swa_initial(spec, fill):
    """Keep the original active fixture and allocate a neutral physical tail."""
    import torch

    original = spec.create_tensor()
    padded = torch.full((DECODE_GRAPH_CAPACITY, *original.shape[1:]), fill, dtype=original.dtype)
    padded[:original.shape[0]].copy_(original)
    return padded


def build_swa_specs(args):
    """Build SWA attention operands, mHC boundaries, and runtime specifications."""
    from golden import ScalarSpec, TensorSpec
    import torch
    from functools import partial
    from config import MAX_BATCH_PER_DP
    from utils import packed_query_lengths
    from sparse_attention import SWA_INPUT_NAMES, make_swa_inputs, make_packed_swa_inputs

    if getattr(args, "dp", 1) != 1:
        raise ValueError("decode SWA specs describe one device")
    packed = getattr(args, "start_pos", None) is not None or args.tokens > MAX_BATCH_PER_DP
    packed = packed or args.requests != args.tokens or args.case in ("prefix", "shuffle")
    if packed:
        query_lens = getattr(args, "query_lens", None)
        lengths = packed_query_lengths(args.tokens, args.requests, query_lens)
        if query_lens is None and args.requests > 1 and lengths[-1] > 16:
            lengths[0] += 7
            lengths[-1] -= 7
        prefixes = ([127, 128, 511, 1048000] * 8)[:args.requests] if args.case == "prefix" else [0] * args.requests
        if getattr(args, "start_pos", None) is not None:
            prefixes = args.start_pos
        pages = 1 + sum((p + n + 128) // 128 - max(0, p - 127) // 128 + 1 for p, n in zip(prefixes, lengths))
    else:
        pages = args.tokens + 1
    leaf_shapes = (
        [args.tokens, D], [D, Q_LORA], [D // 32, Q_LORA], [Q_LORA],
        [Q_LORA, LOCAL_H * HEAD_DIM], [Q_LORA // 32, LOCAL_H * HEAD_DIM],
        [D, HEAD_DIM], [D // 32, HEAD_DIM], [HEAD_DIM], [LOCAL_H],
        [LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], [LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA],
        [LOCAL_O_WIDTH, D], [LOCAL_O_WIDTH // 32, D],
        [args.tokens, ROPE_DIM // 2], [args.tokens, ROPE_DIM // 2], [args.tokens], [args.tokens, 128],
        [pages, 128, 1, HEAD_DIM], [pages, 128, 1, HEAD_DIM // 32],
    )
    bf, fp, mx = torch.bfloat16, torch.float8_e4m3fn, torch.float8_e8m0fnu
    dtypes = (bf, fp, mx, bf, fp, mx, fp, mx, bf, torch.float32, fp, mx, fp, mx,
              torch.float32, torch.float32, torch.int64, torch.int32, fp, bf)
    values = None
    if not getattr(args, "compile_only", False) and not getattr(args, "golden_data", None):
        if packed:
            values = make_packed_swa_inputs(
                args.tokens, args.requests, args.seed, args.case,
                start_pos=getattr(args, "start_pos", None), query_lens=getattr(args, "query_lens", None),
            )
        else:
            values = make_swa_inputs(args.tokens, args.seed)
            if args.case == "masked":
                values["window_slots"].fill_(-1)
                values["window_indices"].fill_(-1)
            elif args.case == "zero":
                values["x"].zero_()
                values["window_cache"].view(torch.uint8).zero_()
            elif args.case == "sink":
                values["attn_sink"].fill_(1000)
    specs = []
    for name, shape, dtype in zip(SWA_INPUT_NAMES, leaf_shapes, dtypes):
        if name == "x":
            continue
        initialize = None if values is None else partial(torch.clone, values[name])
        specs.append(TensorSpec(name, shape, dtype, init_value=initialize))
    local = getattr(args, "local_tokens", args.tokens)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    shapes = {
        "x_hc": [local, HC_MULT, D],
        "incoming_pre_mix": [local, HC_MULT],
        "hc_attn_fn": [MIX_HC, HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [MIX_HC],
        "attn_norm_weight": [D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        if name in ("x_hc", "incoming_pre_mix"):
            stacked = torch.randn(shape, generator=generator)
            if name == "incoming_pre_mix":
                stacked = torch.softmax(stacked, dim=-1)
        else:
            stacked = value
        specs.append(TensorSpec(name, shape, value.dtype, init_value=stacked))
    for name, shape, dtype in (
        ("attention_output", [local, D], torch.bfloat16),
        ("attention_hidden", [local, HC_MULT, D], torch.float32),
        ("attention_pre_mix", [local, HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name == "attention_output" else 0.0
        specs.append(TensorSpec(name, shape, dtype, init_value=sentinel))
    specs.append(TensorSpec("gathered", [args.tokens, D], torch.bfloat16, init_value=13.0))
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(ScalarSpec("attention_epoch", torch.int32, 1, compile_runtime=True, benchmark_step=None))
    by_name = {spec.name: spec for spec in specs}
    from dataclasses import replace

    padding = {"x_hc": 0, "incoming_pre_mix": 0, "rope_cos": 1, "rope_sin": 0,
               "window_slots": -1, "window_indices": -1, "attention_output": 13.0,
               "attention_hidden": 0.0, "attention_pre_mix": 0.0}
    for name, fill in padding.items():
        spec = by_name[name]
        if not 0 <= spec.shape[0] <= DECODE_GRAPH_CAPACITY:
            raise ValueError(f"{name} exceeds physical SWA decode capacity")
        by_name[name] = replace(spec, shape=[DECODE_GRAPH_CAPACITY, *spec.shape[1:]],
                                init_value=partial(_padded_swa_initial, spec, fill))
    return [by_name[name] for name in l3_decode_swa.param_names]


def golden_swa(tensors, *, reference=None):
    """Compose TP1 mHC, normalization, QKV, attention and output references."""
    import torch
    from sparse_attention import SWA_INPUT_NAMES, swa_official_reference
    from hc import golden_hc_collapse, golden_mhc_mixes, golden_hc_post
    from rmsnorm import golden_rms_norm

    active = int(tensors["num_tokens"])
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"],
        tensors["hc_attn_fn"], tensors["hc_attn_scale"], tensors["hc_attn_base"], hf32=True,
    )
    tensors["attention_pre_mix"].copy_(pre_mix)
    collapsed = golden_hc_collapse(tensors["x_hc"], tensors["incoming_pre_mix"]).to(torch.bfloat16)
    normalized = golden_rms_norm(collapsed[:active], tensors["attn_norm_weight"])
    if reference is None:
        inputs = {name: normalized if name == "x" else tensors[name] for name in SWA_INPUT_NAMES}
        for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
            inputs[name] = inputs[name][:active]
        cache_rows = tensors["window_cache"].shape[0] * 128
        shared_cache = tensors["attn_sink"].numel() != 64 and active >= 32 and cache_rows <= active * 128
        partial, cache, scale = swa_official_reference(inputs, shared_cache=shared_cache)
        tensors["attention_output"][:active].copy_(partial.bfloat16())
        tensors["window_cache"].view(torch.uint8).copy_(cache.contiguous().view(torch.uint8))
        tensors["window_cache_scale"].copy_(scale)
    else:
        import inspect

        parameters = inspect.signature(reference).parameters
        inputs = {name: normalized if name == "x" else tensors[name] for name in parameters}
        for name in ("rope_cos", "rope_sin", "window_slots", "window_indices"):
            if name in inputs:
                inputs[name] = inputs[name][:active]
        computed = reference(**inputs)
        tensors["attention_output"][:active].copy_(computed["output"])
        for name in ("window_cache", "window_cache_scale"):
            tensors[name].view(torch.uint8).copy_(computed[name].contiguous().view(torch.uint8))
    tensors["attention_hidden"].zero_()
    tensors["attention_output"][active:].zero_()
    if active:
        hidden = golden_hc_post(
            tensors["attention_output"][:active], tensors["x_hc"][:active],
            post_mix[:active], residual_mix[:active],
        )
        tensors["attention_hidden"][:active].copy_(hidden)
    tensors["attention_input"] = collapsed


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig
    from golden import run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 TP1 SWA decode composition")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=72, requests=12, startpos=[8192] * 12)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    if args.tokens != 72:
        parser.error("this decode entry requires 72 query tokens")
    args.start_pos = args.startpos
    args.dp = 1
    args.active_tokens = args.tokens
    args.bench = False
    devices = [args.device]
    result = run(
        fn=l3_decode_swa,
        specs=build_swa_specs(args),
        golden_fn=golden_swa,
        compare_fn={
            "attention_hidden": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
            "attention_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache", mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=128, leading_rank_axis=False,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.01,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache scale", mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=128, leading_rank_axis=False,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
        },
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)

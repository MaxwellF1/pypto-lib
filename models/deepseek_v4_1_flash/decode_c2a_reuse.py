# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C2A reuse decode attention with mHC."""


import pypto.language as pl
from qkv_proj_rope import PROJ_M_TILE as QKV_PROJ_M_TILE
from hc import LINEAR_T_TILE as HC_LINEAR_T_TILE, LINEAR_OK as HC_LINEAR_OK, MIX_PAD as HC_MIX_PAD
from config import (
    FLASH as M, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN, COMPRESSED_CACHE_GROUP,
    WINDOW_CACHE_GROUP,
)
from hc import hc_head_norm_decode as hc_head_norm, hc_project_decode as hc_project, hc_sinkhorn_decode as hc_sinkhorn, hc_post_decode as hc_post
from qkv_proj_rope import quantize_projection_input_decode as quantize_projection_input, q_proj_rope_decode as q_proj_rope, kv_proj_decode as kv_proj, window_publish_decode as window_publish
from sparse_attention import _attend_project

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
INDEX_TOPK = M.index_topk
MX_GROUP = M.weight_block_size[0]


from config import MOE_TOKENS as DECODE_GRAPH_CAPACITY


@pl.jit.inline(auto_scope=False)
def decode_c2a_reuse(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    window_scale_scratch: pl.Tensor,
    window_blocks: pl.Scalar[pl.INDEX],
):
    post_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_norm = pl.create_tensor([DECODE_GRAPH_CAPACITY, D], dtype=pl.BF16)
    cache_ready = hc_head_norm(x_hc, pre_mix, attn_norm_weight, x_norm)

    qr = pl.create_tensor([DECODE_GRAPH_CAPACITY, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([DECODE_GRAPH_CAPACITY, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    kv_projection = pl.create_tensor([DECODE_GRAPH_CAPACITY, HEAD_DIM], dtype=pl.BF16)
    qkv_x_norm_rows = ((DECODE_GRAPH_CAPACITY + QKV_PROJ_M_TILE - 1) // QKV_PROJ_M_TILE) * QKV_PROJ_M_TILE
    qkv_x_norm_payload = pl.create_tensor([qkv_x_norm_rows, D], dtype=pl.FP8E4M3FN)
    qkv_x_norm_scales = pl.create_tensor([1, qkv_x_norm_rows * (D // 32)], dtype=pl.FP8E8M0)
    proj_payload, proj_scales = quantize_projection_input(x_norm, num_tokens, qkv_x_norm_payload, qkv_x_norm_scales)
    qb_rows = DECODE_GRAPH_CAPACITY
    qb_workspace_payload = pl.create_tensor([qb_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_workspace_scales = pl.create_tensor([1, qb_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    q_proj_result = q_proj_rope(
        proj_payload, proj_scales,
        wq_a, wq_a_scale, q_norm_weight,
        wq_b, wq_b_scale,
        rope_cos, rope_sin,
        qr, query,
        num_tokens,
     qb_workspace_payload, qb_workspace_scales)
    kv_projection = kv_proj(proj_payload, proj_scales, wkv, wkv_scale, kv_projection, num_tokens)
    window_publish(
        kv_projection, kv_norm_weight, rope_cos, rope_sin,
        window_slots, window_cache, window_cache_scale,
        num_tokens, cache_ready,
     window_scale_scratch, cache_blocks=window_blocks)

    chunk_done = cache_ready

    attn_out = pl.create_tensor([DECODE_GRAPH_CAPACITY, D], dtype=pl.BF16)
    project_attended = pl.create_tensor([DECODE_GRAPH_CAPACITY, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    _attend_project(
        query, window_indices, window_cache, window_cache_scale, compressed_indices,
        compressed_cache, compressed_cache_scale, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, attn_out,
        num_tokens, chunk_done, compressed_keys=INDEX_TOPK,
     attended=project_attended, physical_capacity=DECODE_GRAPH_CAPACITY)

    hc_workspace_rows = ((DECODE_GRAPH_CAPACITY + HC_LINEAR_T_TILE - 1) // HC_LINEAR_T_TILE) * HC_LINEAR_T_TILE
    hc_inv_rms = pl.create_tensor([hc_workspace_rows, 1], dtype=pl.FP32)
    hc_partials = pl.create_tensor([HC_LINEAR_OK * hc_workspace_rows, HC_MIX_PAD], dtype=pl.FP32)
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, hf32=True, inv_rms=hc_inv_rms, mixes_partials=hc_partials)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output, next_pre_mix


@pl.jit
def l2_decode_c2a_reuse(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    window_scale_scratch = pl.create_tensor([pl.tensor.dim(window_cache_scale, 0), 128, 1, HEAD_DIM // 32], dtype=pl.BF16)
    with pl.graph("lib_decode_c2a_reuse"):
        decode_c2a_reuse(
            x_hc, pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale,
            compressed_indices,
            output, next_pre_mix,
            num_tokens,
         window_scale_scratch, window_blocks=pl.tensor.dim(window_cache_scale, 0))
    return output, next_pre_mix


@pl.jit.host
def l3_decode_c2a_reuse(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
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
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    compressed_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32],
    output: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    wq_a_scale_mx: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_mx: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_mx: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_mx: pl.Tensor[[LOCAL_O_WIDTH // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    l2_decode_c2a_reuse(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_mx, q_norm_weight, wq_b, wq_b_scale_mx,
        wkv, wkv_scale_mx, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_mx,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale,
        compressed_indices,
        output, next_pre_mix,
        num_tokens, device=0,
    )


if __name__ == "__main__":
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run
    from decode_c2a_full import build_c2a_specs, golden_c2a
    from functools import partial

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C2A reuse validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["mixed", "long", "masked", "zero"], default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=32, requests=6)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    host = l3_decode_c2a_reuse
    specs = build_c2a_specs(
        "reuse", args.tokens, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens, param_names=host.param_names,
    )
    result = run(
        fn=host,
        specs=specs,
        golden_fn=partial(golden_c2a, mode="reuse"),
        compare_fn={
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache", mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=128, leading_rank_axis=False,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window cache scale", mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=128, leading_rank_axis=False,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            enable_chip_swimlane=args.enable_chip_swimlane,
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
    )
    if not result.passed:
        raise SystemExit(result.error or 1)

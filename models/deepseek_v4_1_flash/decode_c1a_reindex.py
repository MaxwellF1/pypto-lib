# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C1A reindex decode attention with mHC."""

import pypto.language as pl
from decode_indexer import paged_score_output_flag
from decode_indexer import CANDIDATE_BLOCK_SIZE
from decode_indexer import INDEX_WEIGHT_K_SPLIT_TILE as IDX_WEIGHT_SPLITS
from qkv_proj_rope import PROJ_M_TILE as QKV_PROJ_M_TILE
from hc import LINEAR_T_TILE as HC_LINEAR_T_TILE, LINEAR_OK as HC_LINEAR_OK, MIX_PAD as HC_MIX_PAD
from config import (
    COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, WINDOW_CACHE_GROUP, FLASH as M, B_DYN, CMP_BLOCKS_DYN,
    INDEX_BLOCKS_DYN, INDEX_POOL_BLOCKS_DYN, ORI_BLOCKS_DYN, TABLE_DYN,
)
from hc import hc_head_norm_decode as hc_head_norm, hc_project_decode as hc_project, hc_sinkhorn_decode as hc_sinkhorn, hc_post_decode as hc_post
from decode_indexer import c1a_index, score_select_reindex
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
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


from config import MOE_TOKENS as DECODE_GRAPH_CAPACITY


@pl.jit.inline(auto_scope=False)
def decode_c1a_reindex(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16],
    request_ids: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    compressed_lens: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32],
    output: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    x_norm: pl.Tensor,
    qr: pl.Tensor,
    query: pl.Tensor,
    kv_projection: pl.Tensor,
    qkv_x_norm_payload: pl.Tensor,
    qkv_x_norm_scales: pl.Tensor,
    qb_workspace_payload: pl.Tensor,
    qb_workspace_scales: pl.Tensor,
    window_scale_scratch: pl.Tensor,
    c1a_index_query: pl.Tensor,
    c1a_index_query_scale: pl.Tensor,
    c1a_index_weights: pl.Tensor,
    c1a_index_scores: pl.Tensor,
    c1a_weight_partials: pl.Tensor,
    attn_out: pl.Tensor,
    project_attended: pl.Tensor,
    hc_inv_rms: pl.Tensor,
    hc_partials: pl.Tensor,
    reindex_paged_scores: pl.Tensor,
    reindex_score_runs: pl.Tensor,
    reindex_candidate_keys: pl.Tensor,
    reindex_candidate_indices: pl.Tensor,
    pool_payloads: pl.Tensor,
    pool_scales: pl.Tensor,
    pool_valid: pl.Tensor,
    pool_physical: pl.Tensor,
    pool_runs: pl.Tensor,
    pool_candidate_keys: pl.Tensor,
    pool_candidate_indices: pl.Tensor,
    window_blocks: pl.Scalar[pl.INDEX], pool_capacity: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    precomputed_emit_scores: pl.constexpr,
):
    x_ready = hc_head_norm(x_hc, pre_mix, attn_norm_weight, x_norm)
    qkv_x_norm_rows = ((pl.tensor.dim(x_norm, 0) + QKV_PROJ_M_TILE - 1) // QKV_PROJ_M_TILE) * QKV_PROJ_M_TILE
    proj_payload, proj_scales = quantize_projection_input(x_norm, num_tokens, qkv_x_norm_payload, qkv_x_norm_scales)
    qb_rows = pl.tensor.dim(proj_payload, 0)
    qr, query, qb_payload, qb_scales = q_proj_rope(
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
        num_tokens, x_ready,
     window_scale_scratch, cache_blocks=window_blocks)
    score_width = pool_capacity * 8
    index_query, index_query_scale, index_weights, scores, query_tid, weights_tid = c1a_index(
        x_norm, index_wq_b, index_wq_b_scale, index_weights_proj, rope_cos, rope_sin,
        num_tokens, score_width, x_ready,
        qb_payload, qb_scales,
     c1a_index_query, c1a_index_query_scale, c1a_index_weights, c1a_index_scores, c1a_weight_partials)
    selection_ready = score_select_reindex(
        index_query, index_query_scale, index_weights, index_cache, index_cache_scale,
        index_block_table, request_ids, compressed_lens, pool_block_ids, pool_block_counts,
        scores, topk_indices, num_tokens, x_ready, query_tid, weights_tid, x_ready,
     reindex_paged_scores, reindex_score_runs, reindex_candidate_keys, reindex_candidate_indices, pool_payloads, pool_scales, pool_valid, pool_physical, pool_runs, pool_candidate_keys, pool_candidate_indices, pool_capacity=pool_capacity, emit_scores=emit_scores, precomputed_emit_scores=precomputed_emit_scores)
    _attend_project(
        query, window_indices, window_cache, window_cache_scale, topk_indices,
        compressed_cache, compressed_cache_scale, attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, attn_out,
        num_tokens, selection_ready, compressed_keys=INDEX_TOPK,
     attended=project_attended, physical_capacity=DECODE_GRAPH_CAPACITY)
    hc_workspace_rows = ((pl.tensor.dim(x_hc, 0) + HC_LINEAR_T_TILE - 1) // HC_LINEAR_T_TILE) * HC_LINEAR_T_TILE
    inv_rms, partials = hc_project(x_hc, hc_attn_fn, hf32=True, inv_rms=hc_inv_rms, mixes_partials=hc_partials)
    hc_sinkhorn(inv_rms, partials, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_decode_c1a_reindex(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    compressed_lens: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    qkv_x_norm_rows = ((DECODE_GRAPH_CAPACITY + QKV_PROJ_M_TILE - 1) // QKV_PROJ_M_TILE) * QKV_PROJ_M_TILE
    qb_rows = qkv_x_norm_rows
    hc_workspace_rows = ((DECODE_GRAPH_CAPACITY + HC_LINEAR_T_TILE - 1) // HC_LINEAR_T_TILE) * HC_LINEAR_T_TILE
    score_width = pl.tensor.dim(pool_block_ids, 1) * CANDIDATE_BLOCK_SIZE
    reindex_post_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT], dtype=pl.FP32)
    reindex_residual_mix = pl.create_tensor([DECODE_GRAPH_CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
    reindex_x_norm = pl.create_tensor([DECODE_GRAPH_CAPACITY, D], dtype=pl.BF16)
    reindex_qr = pl.create_tensor([DECODE_GRAPH_CAPACITY, Q_LORA], dtype=pl.BF16)
    reindex_query = pl.create_tensor([DECODE_GRAPH_CAPACITY, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    reindex_kv_projection = pl.create_tensor([DECODE_GRAPH_CAPACITY, HEAD_DIM], dtype=pl.BF16)
    reindex_qkv_x_norm_payload = pl.create_tensor([qkv_x_norm_rows, D], dtype=pl.FP8E4M3FN)
    reindex_qkv_x_norm_scales = pl.create_tensor([1, qkv_x_norm_rows * (D // 32)], dtype=pl.FP8E8M0)
    reindex_qb_workspace_payload = pl.create_tensor([qb_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    reindex_qb_workspace_scales = pl.create_tensor([1, qb_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    reindex_window_scale_scratch = pl.create_tensor([pl.tensor.dim(window_cache_scale, 0), 128, 1, HEAD_DIM // 32], dtype=pl.BF16)
    reindex_c1a_index_query = pl.create_tensor([DECODE_GRAPH_CAPACITY, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    reindex_c1a_index_query_scale = pl.create_tensor([DECODE_GRAPH_CAPACITY, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0)
    reindex_c1a_index_weights = pl.create_tensor([DECODE_GRAPH_CAPACITY, INDEX_H], dtype=pl.BF16)
    reindex_c1a_index_scores = pl.create_tensor([DECODE_GRAPH_CAPACITY, score_width], dtype=pl.FP32)
    reindex_c1a_weight_partials = pl.create_tensor([IDX_WEIGHT_SPLITS * DECODE_GRAPH_CAPACITY, INDEX_H], dtype=pl.FP32)
    reindex_attn_out = pl.create_tensor([DECODE_GRAPH_CAPACITY, D], dtype=pl.BF16)
    reindex_project_attended = pl.create_tensor([DECODE_GRAPH_CAPACITY, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    reindex_hc_inv_rms = pl.create_tensor([hc_workspace_rows, 1], dtype=pl.FP32)
    reindex_hc_partials = pl.create_tensor([HC_LINEAR_OK * hc_workspace_rows, HC_MIX_PAD], dtype=pl.FP32)
    reindex_score_width = score_width
    reindex_table_width = pl.tensor.dim(index_block_table, 1)
    reindex_dense_width = pl.min(reindex_score_width, reindex_table_width * 128)
    reindex_panels = (reindex_dense_width + 255) // 256
    reindex_workers = 32
    reindex_shards = reindex_workers
    if reindex_panels > 100:
        reindex_shards = pl.max(reindex_workers, (DECODE_GRAPH_CAPACITY * reindex_panels + 99) // 100)
    reindex_paged_scores = pl.create_tensor([DECODE_GRAPH_CAPACITY, reindex_dense_width], dtype=pl.FP32)
    reindex_score_runs = pl.create_tensor([4, DECODE_GRAPH_CAPACITY + 1], dtype=pl.INT32)
    reindex_candidate_keys = pl.create_tensor([DECODE_GRAPH_CAPACITY, reindex_shards * INDEX_TOPK], dtype=pl.UINT16)
    reindex_candidate_indices = pl.create_tensor([DECODE_GRAPH_CAPACITY, reindex_shards * INDEX_TOPK], dtype=pl.INT32)
    pool_width = score_width
    pool_panels = (pool_width + 255) // 256
    pool_total_rows = DECODE_GRAPH_CAPACITY * pool_panels * 256
    pool_workers = 32
    pool_payloads = pl.create_tensor([pool_total_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    pool_scales = pl.create_tensor([1, pool_total_rows * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.UINT8)
    pool_valid = pl.create_tensor([1, pool_total_rows], dtype=pl.INT32)
    pool_physical = pl.create_tensor([1, pool_total_rows], dtype=pl.INT32)
    pool_runs = pl.create_tensor([4, DECODE_GRAPH_CAPACITY + 1], dtype=pl.INT32)
    pool_candidate_keys = pl.create_tensor([DECODE_GRAPH_CAPACITY, pool_workers * INDEX_TOPK], dtype=pl.UINT16)
    pool_candidate_indices = pl.create_tensor([DECODE_GRAPH_CAPACITY, pool_workers * INDEX_TOPK], dtype=pl.INT32)
    # Bind the descriptor-derived flag as a Graph argument for each dispatch.
    emit_scores = paged_score_output_flag(pl.tensor.dim(reindex_paged_scores, 1), pl.tensor.dim(pool_block_ids, 1))
    with pl.graph("c1a_reindex"):
        result = decode_c1a_reindex(
            x_hc, pre_mix,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_weight,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight,
            attn_sink,
            wo_a, wo_a_scale, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices,
            window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
            request_ids, compressed_lens,
            index_cache, index_cache_scale, index_block_table,
            pool_block_ids, pool_block_counts,
            index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices,
            output, next_pre_mix,
            num_tokens,
         reindex_post_mix, reindex_residual_mix, reindex_x_norm, reindex_qr, reindex_query, reindex_kv_projection, reindex_qkv_x_norm_payload, reindex_qkv_x_norm_scales, reindex_qb_workspace_payload, reindex_qb_workspace_scales, reindex_window_scale_scratch, reindex_c1a_index_query, reindex_c1a_index_query_scale, reindex_c1a_index_weights, reindex_c1a_index_scores, reindex_c1a_weight_partials, reindex_attn_out, reindex_project_attended, reindex_hc_inv_rms, reindex_hc_partials, reindex_paged_scores, reindex_score_runs, reindex_candidate_keys, reindex_candidate_indices, pool_payloads, pool_scales, pool_valid, pool_physical, pool_runs, pool_candidate_keys, pool_candidate_indices, window_blocks=pl.tensor.dim(window_cache_scale, 0), pool_capacity=pl.tensor.dim(pool_block_ids, 1), emit_scores=emit_scores, precomputed_emit_scores=True)
    return result


@pl.jit.host
def l3_decode_c1a_reindex(
    x_hc: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[DECODE_GRAPH_CAPACITY, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[DECODE_GRAPH_CAPACITY, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    request_ids: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    compressed_lens: pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    pool_block_ids: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[DECODE_GRAPH_CAPACITY], pl.INT32]],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, INDEX_TOPK], pl.INT32]],
    output: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT, D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[DECODE_GRAPH_CAPACITY, HC_MULT], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    # MX_B_NN scale views.
    wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale
    wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale
    wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale
    wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale
    index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale
    l2_decode_c1a_reindex(
        x_hc, pre_mix,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale_r, q_norm_weight,
        wq_b, wq_b_scale_r,
        wkv, wkv_scale_r, kv_norm_weight,
        attn_sink,
        wo_a, wo_a_scale, wo_b, wo_b_scale_r,
        rope_cos, rope_sin,
        window_slots, window_indices,
        window_cache, window_cache_scale, compressed_cache, compressed_cache_scale,
        request_ids, compressed_lens,
        index_cache, index_cache_scale, index_block_table,
        pool_block_ids, pool_block_counts,
        index_wq_b, index_wq_b_scale_r, index_weights_proj,
        topk_indices,
        output, next_pre_mix,
        num_tokens,
        device=0,
    )


if __name__ == "__main__":
    from functools import partial
    from config import BLOCK_SIZE
    from decode_c1a_full import build_c1a_specs, golden_c1a
    from golden import mapped_pool_ratio_allclose, ratio_allclose, run

    from utils import configure_decode_workload, parse_decode_query_lens, parse_decode_startpos

    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="TP1 A5 mHC C1A reindex validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    startpos_help = "one starting position per request; omitted: default validation workload"
    parser.add_argument("--start-pos", "--startpos", dest="startpos", type=parse_decode_startpos, help=startpos_help)
    query_help = "one query length per start-pos; omitted: config.DECODE_ROWS_PER_REQUEST each"
    parser.add_argument("--query-lens", type=parse_decode_query_lens, help=query_help)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=["random", "ragged", "masked", "sink"], default="random")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.set_defaults(tokens=1, requests=1, pages=8)
    args = parser.parse_args()
    configure_decode_workload(args, parser)
    if args.startpos is not None:
        max_end = max(pos + length for pos, length in zip(args.startpos, args.query_lens))
        args.pages = (max_end + 127) // 128
    host = l3_decode_c1a_reindex
    specs = build_c1a_specs(
        "reindex", args.tokens, args.pages, args.requests, args.seed, args.case,
        start_pos=args.startpos, query_lens=args.query_lens, param_names=host.param_names,
    )

    result = run(
        fn=host,
        specs=specs,
        golden_fn=partial(golden_c1a, mode="reindex"),
        compile_only=args.compile_only,
        save_data=args.save_data,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            dump_passes=args.dump_passes,
            enable_chip_swimlane=args.enable_chip_swimlane,
            distributed_config=DistributedConfig(device_ids=[args.device], num_sub_workers=0),
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "topk_indices": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.02),
            "pool_block_ids": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "pool_block_counts": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "output": ratio_allclose(atol=1e-2, rtol=1e-2, max_error_ratio=0.01),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache",
                mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=2 ** -9, rtol=2 ** -3, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache_scale",
                mapping_shape=(DECODE_GRAPH_CAPACITY,),
                block_size=BLOCK_SIZE, leading_rank_axis=False,
                atol=0.0, rtol=2 ** -7, max_error_ratio=0.0,
            ),
            "compressed_cache": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "compressed_cache_scale": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "index_cache": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            "index_cache_scale": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)

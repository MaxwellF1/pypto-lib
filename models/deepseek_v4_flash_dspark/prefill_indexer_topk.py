# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 exact prefill indexer top-k selection."""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, FP32_NEG_INF, PREFILL_MAX_CONTEXT_TOKENS


# model config
COMPRESS_RATIO = 4
IDX_N_HEADS = M.index_n_heads
IDX_HEAD_DIM = M.index_head_dim
IDX_TOPK = M.index_topk
INDEXER_MAX_RAW_TOKENS = PREFILL_MAX_CONTEXT_TOKENS
INDEXER_MAX_CANDIDATES = INDEXER_MAX_RAW_TOKENS // COMPRESS_RATIO
TOPK_PAIR_WIDTH = 2 * IDX_TOPK

# tiling
SELECTOR_QUERY_TILE = 128
TOPK_LEAF_TILE = 8192
TOPK_GROUP_TILE = 2

# workspace geometry
TOPK_MAX_LEAVES = INDEXER_MAX_CANDIDATES // TOPK_LEAF_TILE
TOPK_GROUPS_PER_QUERY = TOPK_MAX_LEAVES // TOPK_GROUP_TILE
TOPK_GROUP_WORKERS = 48
TOPK_GROUP_ROOT_ROWS = SELECTOR_QUERY_TILE * TOPK_GROUPS_PER_QUERY
TOPK_GROUP_SCRATCH_ROWS = TOPK_GROUP_WORKERS * TOPK_GROUP_TILE
TOPK_ARENA_ROWS = TOPK_GROUP_ROOT_ROWS + TOPK_GROUP_SCRATCH_ROWS
TOPK_SCORE_WORKERS = 24


@pl.jit.inline
def _merge2_top512_pairs(
    pair_arena: pl.Tensor,
    left_slot: pl.Scalar[pl.INDEX],
    right_slot: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    left = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
    right = pl.load(pair_arena, [right_slot, 0], [1, TOPK_PAIR_WIDTH])
    merge_tmp = pl.tile.create([1, 2 * TOPK_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left, right, tmp=merge_tmp)
    merged = pl.tile.slice(merged_all, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output_slot, 0], pair_arena)


@pl.jit.inline
def _merge_topk_level_pairs(
    pair_arena: pl.Tensor,
    arena_base: pl.Scalar[pl.INDEX],
    input_count: pl.Scalar[pl.INDEX],
    input_base: pl.Scalar[pl.INDEX],
    output_base: pl.Scalar[pl.INDEX],
) -> None:
    output_count = (input_count + 1) // 2
    for output in pl.range(output_count):
        left_slot = arena_base + input_base + 2 * output
        right_slot = left_slot + 1
        output_slot = arena_base + output_base + output
        if right_slot < arena_base + input_base + input_count:
            _merge2_top512_pairs(pair_arena, left_slot, right_slot, output_slot)
        else:
            forwarded = pl.load(pair_arena, [left_slot, 0], [1, TOPK_PAIR_WIDTH])
            pl.store(forwarded, [output_slot, 0], pair_arena)


@pl.jit.inline
def _topk_leaf(
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    query: pl.Scalar[pl.INDEX],
    logical_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
) -> None:
    logical_begin_i32 = pl.cast(logical_begin, pl.INT32)
    leaf_index_ramp = pl.tile.arange(0, [1, TOPK_LEAF_TILE], dtype=pl.INT32)
    leaf_indices = pl.add(leaf_index_ramp, logical_begin_i32)
    leaf_scores_raw = pl.load(score_arena, [query, logical_begin], [1, TOPK_LEAF_TILE], valid_shape=[1, valid_count])
    leaf_scores = pl.tile.fillpad(leaf_scores_raw, pad_value=pl.PadValue.min)
    leaf_floor = pl.tile.full([1, TOPK_LEAF_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
    leaf_scores = pl.maximum(leaf_scores, leaf_floor)
    pairs = pl.tile.sort32(leaf_scores, pl.reinterpret_view(leaf_indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, TOPK_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], pair_arena)


@pl.jit.incore
def _topk_group_wave(
    position_ids: pl.Tensor,
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Reduce striped two-leaf subtrees into compact roots."""
    worker = pl.tile.get_block_idx()
    global_group_base = 0
    for query in pl.range(tile_rows):
        position = pl.read(position_ids, [tile_base + query])
        visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
        leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
        group_count = (leaf_count + TOPK_GROUP_TILE - 1) // TOPK_GROUP_TILE
        base_mod = global_group_base % TOPK_GROUP_WORKERS
        first_group = (worker + base_mod) % TOPK_GROUP_WORKERS
        for group in pl.range(first_group, group_count, TOPK_GROUP_WORKERS):
            leaf_begin = group * TOPK_GROUP_TILE
            group_leaf_count = pl.min(TOPK_GROUP_TILE, leaf_count - leaf_begin)
            group_root_slot = query * TOPK_GROUPS_PER_QUERY + group
            if group_leaf_count == 1:
                logical_begin = leaf_begin * TOPK_LEAF_TILE
                valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                _topk_leaf(
                    score_arena, pair_arena,
                    query, logical_begin, valid_count,
                    group_root_slot,
                )
            else:
                scratch_base = TOPK_GROUP_ROOT_ROWS + worker * TOPK_GROUP_TILE
                for group_leaf in pl.unroll(TOPK_GROUP_TILE):
                    leaf = leaf_begin + group_leaf
                    logical_begin = leaf * TOPK_LEAF_TILE
                    valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                    _topk_leaf(
                        score_arena, pair_arena,
                        query, logical_begin, valid_count,
                        scratch_base + group_leaf,
                    )
                _merge2_top512_pairs(pair_arena, scratch_base, scratch_base + 1, group_root_slot)
        global_group_base = global_group_base + group_count


@pl.jit.incore
def _topk_query_merge(
    position_ids: pl.Tensor,
    pair_arena: pl.Tensor,
    topk_indices: pl.Tensor,
    tile_base: pl.Scalar[pl.INDEX],
):
    """Merge compact group roots into one Top-512 row."""
    query = pl.tile.get_block_idx()
    output_query = tile_base + query
    position = pl.read(position_ids, [output_query])
    visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
    empty_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
    pl.store(empty_indices, [output_query, 0], topk_indices)

    if visible_count > 0:
        leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
        group_count = (leaf_count + TOPK_GROUP_TILE - 1) // TOPK_GROUP_TILE
        arena_base = query * TOPK_GROUPS_PER_QUERY
        if group_count > 1:
            level1_count = (group_count + 1) // 2
            _merge_topk_level_pairs(pair_arena, arena_base, group_count, 0, 0)
            if level1_count > 1:
                level2_count = (level1_count + 1) // 2
                _merge_topk_level_pairs(pair_arena, arena_base, level1_count, 0, 0)
                if level2_count > 1:
                    level3_count = (level2_count + 1) // 2
                    _merge_topk_level_pairs(pair_arena, arena_base, level2_count, 0, 0)
                    if level3_count > 1:
                        _merge_topk_level_pairs(pair_arena, arena_base, level3_count, 0, 0)

        root_pairs = pl.load(pair_arena, [arena_base, 0], [1, TOPK_PAIR_WIDTH])
        root_indices = pl.tile.gather_mask(root_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
        output_indices = pl.tile.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
        valid_topk = pl.min(visible_count, IDX_TOPK)
        for lane in pl.range(valid_topk):
            output_index = pl.tile.read(root_indices, [0, lane])
            pl.tile.write(output_indices, [0, lane], output_index)
        pl.store(output_indices, [output_query, 0], topk_indices)


@pl.jit.inline(auto_scope=False)
def prefill_indexer_score_topk(
    qr_hadamard_i8: pl.Tensor,
    qr_hadamard_scale_dq: pl.Tensor,
    weights: pl.Tensor,
    idx_kv_cache: pl.Tensor,
    idx_kv_scale: pl.Tensor,
    idx_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    topk_indices: pl.Tensor,
    score_arena: pl.Tensor,
    pair_arena: pl.Tensor,
    completion: pl.Array[1, pl.TASK_ID],
    tile_base: pl.Scalar[pl.INDEX],
    tile_rows: pl.Scalar[pl.INDEX],
):
    """Score one query tile and reduce its Top-K forest."""
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    idx_cache_rows = idx_block_num * BLOCK_SIZE
    kv_cache_i8_flat = pl.reshape(idx_kv_cache, [idx_cache_rows, IDX_HEAD_DIM])
    kv_scale_flat = pl.reshape(idx_kv_scale, [idx_cache_rows, 1])
    with pl.spmd(
        TOPK_SCORE_WORKERS, name_hint="prefill_idx_score_leaf_wave", deps=[completion[0]],
        optimizations=[pl.split(pl.SplitMode.NONE, slot_num=2)],
    ) as score_tid:
        worker = pl.tile.get_block_idx()
        global_leaf_base = 0
        for query in pl.range(tile_rows):
            output_query = tile_base + query
            position = pl.read(position_ids, [output_query])
            visible_count = pl.max(pl.min((position + 1) // COMPRESS_RATIO, INDEXER_MAX_CANDIDATES), 0)
            leaf_count = (visible_count + TOPK_LEAF_TILE - 1) // TOPK_LEAF_TILE
            base_mod = global_leaf_base % TOPK_SCORE_WORKERS
            first_leaf = (worker + base_mod) % TOPK_SCORE_WORKERS
            for leaf in pl.range(first_leaf, leaf_count, TOPK_SCORE_WORKERS):
                logical_begin = leaf * TOPK_LEAF_TILE
                valid_count = pl.min(TOPK_LEAF_TILE, visible_count - logical_begin)
                query_head_begin = query * IDX_N_HEADS
                query_vector = qr_hadamard_i8[query_head_begin : query_head_begin + IDX_N_HEADS, 0:IDX_HEAD_DIM]
                query_scale_heads = qr_hadamard_scale_dq[query_head_begin : query_head_begin + IDX_N_HEADS, 0:1]
                query_scale = pl.reshape(query_scale_heads, [1, IDX_N_HEADS])
                query_weight = weights[query : query + 1, 0:IDX_N_HEADS]
                for page in pl.pipeline(0, (valid_count + BLOCK_SIZE - 1) // BLOCK_SIZE, stage=2):
                    page_begin = page * BLOCK_SIZE
                    logical_row = logical_begin + page_begin
                    logical_page = logical_row // BLOCK_SIZE
                    physical_block_raw = pl.read(idx_block_table, [logical_page])
                    score_valid = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=FP32_NEG_INF)
                    if physical_block_raw >= 0 and physical_block_raw < idx_block_num:
                        physical_block = pl.cast(physical_block_raw, pl.INDEX)
                        physical_row = physical_block * BLOCK_SIZE
                        kv_i8 = kv_cache_i8_flat[physical_row : physical_row + BLOCK_SIZE, 0:IDX_HEAD_DIM]
                        score_i32 = pl.matmul(kv_i8, query_vector, out_dtype=pl.INT32, b_trans=True)
                        score_fp32 = pl.cast(score_i32, target_type=pl.FP32, mode="none")
                        score_fp32 = pl.col_expand_mul(score_fp32, query_scale)
                        score_fp32 = pl.maximum(score_fp32, 0.0)
                        score_fp32 = pl.col_expand_mul(score_fp32, query_weight)
                        kv_scale = kv_scale_flat[physical_row : physical_row + BLOCK_SIZE, 0:1]
                        score_sum = pl.row_sum(score_fp32)
                        score_scaled = pl.mul(score_sum, kv_scale)
                        score_row = pl.reshape(score_scaled, [1, BLOCK_SIZE])
                        valid_rows = pl.min(BLOCK_SIZE, valid_count - page_begin)
                        score_valid_view = pl.set_validshape(score_row, 1, valid_rows)
                        score_padded = pl.fillpad(score_valid_view, pad_value=pl.PadValue.min)
                        score_floor = pl.full([1, BLOCK_SIZE], dtype=pl.FP32, value=FP32_NEG_INF)
                        score_valid = pl.maximum(score_padded, score_floor)
                    score_arena[query : query + 1, logical_row : logical_row + BLOCK_SIZE] = score_valid
            global_leaf_base = global_leaf_base + leaf_count

    with pl.spmd(TOPK_GROUP_WORKERS, name_hint="prefill_idx_topk_group_wave", deps=[score_tid]) as topk_tid:
        _topk_group_wave(position_ids, score_arena, pair_arena, tile_base, tile_rows)

    with pl.spmd(tile_rows, name_hint="prefill_idx_topk_query_merge", deps=[topk_tid]) as merge_tid:
        _topk_query_merge(position_ids, pair_arena, topk_indices, tile_base)

    completion[0] = merge_tid
    return topk_indices

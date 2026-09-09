# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4 context-parallel attention and MoE stages."""

import pypto.language as pl
import pypto.language.distributed as pld


# The production CP path owns 2 parts * 4 attention tiles * 128 rows on every
# rank. Its MoE capacity is selected explicitly after importing shared
# primitives; the attention leaves retain their independent 128-row tile ABI.
import config
_EXPECTED_CP_LOCAL_ROWS = 2 * 4 * 128
FWD_RAW_BLOCKS_DYN = pl.dynamic("FWD_RAW_BLOCKS_DYN")
CSA_FWD_IDX_BLOCKS_DYN = pl.dynamic("CSA_FWD_IDX_BLOCKS_DYN")
CSA_FWD_STATE_BLOCKS_DYN = pl.dynamic("CSA_FWD_STATE_BLOCKS_DYN")
CSA_FWD_INNER_STATE_BLOCKS_DYN = pl.dynamic("CSA_FWD_INNER_STATE_BLOCKS_DYN")
HCA_FWD_STATE_BLOCKS_DYN = pl.dynamic("HCA_FWD_STATE_BLOCKS_DYN")
HCA_FWD_CMP_BLOCKS_DYN = pl.dynamic("HCA_FWD_CMP_BLOCKS_DYN")
CSA_FWD_CMP_BLOCKS_DYN = pl.dynamic("CSA_FWD_CMP_BLOCKS_DYN")

from moe import (
    PrefillMoELayout, make_prefill_moe, clear_prefill_moe_signals, check_prefill_moe_slab,
    PREFILL_MOE_EXPERT_SCALE_PAD, PREFILL_MOE_SCALE_PAD, D, HC_DIM, HC_MULT, MIX_HC, MOE_INTER,
    N_EXPERTS_GLOBAL, N_LOCAL, N_RANKS, TOPK, VOCAB,
)

MOE_ROWS = _EXPECTED_CP_LOCAL_ROWS
PREFILL_MOE_LAYOUT = PrefillMoELayout(MOE_ROWS)
prefill_moe = make_prefill_moe(PREFILL_MOE_LAYOUT)
PREFILL_MOE_ROUTES_PER_SRC = PREFILL_MOE_LAYOUT.routes_per_source
PREFILL_MOE_TOTAL_CAP = PREFILL_MOE_LAYOUT.total_capacity
PREFILL_MOE_GROUPED_TOTAL_CAP = PREFILL_MOE_LAYOUT.grouped_capacity
from prefill_swa import (
    BLOCK_ROWS, CP_CHOICES, CP_SIZE, CP_TAIL_WINDOW_ROWS, H, HEAD_DIM, LOCAL_PARTS, MAX_SEQ_LEN, NUM_SEGMENTS,
    O_GROUPS, O_GROUP_IN, O_LORA, ORI_MAX_BLOCKS, OVERLAY_BASE, OVERLAY_ROWS, OVERLAY_SOURCES, Q_LORA,
    ROPE_HEAD_DIM, TAIL_ROWS, WIN, prefill_cp_swa_core,
)
from prefill_cp_zigzag import MAX_SEGMENT_TILES
from prefill_cp_zigzag import CP_PREFILL_CMP_BLOCK_NUM as PREFILL_CMP_BLOCK_NUM
# HCA / CSA inline cores and their type-specific constants. The FWD child
# calls the cores directly (never @pl.jit children); the constants are used
# only for static child-side shape annotations and typed pl.slice offsets.
from prefill_hca import (
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE, COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM, HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, IDX_TOPK,
    prefill_cp_hca_core,
)
from prefill_csa import (
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE, COMPRESS_RATIO as CSA_COMPRESS_RATIO, IDX_HEAD_DIM,
    IDX_N_HEADS, INNER_OUT_DIM as CSA_INNER_OUT_DIM, INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM as CSA_INNER_STATE_DIM, INNER_STATE_MAX_BLOCKS as CSA_INNER_STATE_MAX_BLOCKS,
    LOCAL_LEAVES as CSA_LOCAL_LEAVES, MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM as CSA_MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS as CSA_MAIN_STATE_MAX_BLOCKS, MAX_COMPRESS_LEAVES as CSA_MAX_COMPRESS_LEAVES,
    prefill_cp_csa_core,
)
from config import IDX_CACHE_MAX_BLOCKS, PREFILL_CMP_MAX_BLOCKS
from prefill_cp_exchange import _clear_prefill_cp_exchange_signals
from prefill_cp_exchange import (
    CMP_META_DIM, CMP_WINDOW_ROWS, META_DIM, RECORDS_PER_WINDOW, SCALE_TILE_COLS, STATE_META_DIM,
    STATE_RECORDS_PER_WINDOW, STATE_WINDOW_ROWS,
)
# Final normalization uses the shared HC head and RMSNorm kernels.
# The canonical HOST entry owns the LM head.
from hc_head import hc_head
from rmsnorm import rms_norm

# ---------------------------------------------------------------------------
# Static CP/EP contract
# ---------------------------------------------------------------------------
# This entry builds the compact prefill MoE, whose tiles constrain the slab.
check_prefill_moe_slab(MOE_ROWS)
assert CP_SIZE in CP_CHOICES, f"--cp must be one of {CP_CHOICES} (got {CP_SIZE})"
assert CP_SIZE in (1, N_RANKS), (
    f"Prefill requires CP=1 or CP=world, with EP=world (got CP={CP_SIZE}, EP={N_RANKS})"
)
assert LOCAL_PARTS == 2
ATTN_TILE_ROWS = TAIL_ROWS
NUM_ATTN_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_ATTN_TILES * ATTN_TILE_ROWS
assert ATTN_TILE_ROWS == 128, (f"CP attention leaf ABI requires 128 rows (got {ATTN_TILE_ROWS})")
assert LOCAL_ROWS == _EXPECTED_CP_LOCAL_ROWS == MOE_ROWS, (
    f"production CP MoE requires one {LOCAL_ROWS}-row local slab "
    f"(configured MOE_ROWS={MOE_ROWS})"
)
# Model order: SWA layers 0/1, then alternating CSA/HCA with final CSA.
FWD_NUM_LAYERS = config.FLASH.num_hidden_layers
HCA_NUM_LAYERS = (FWD_NUM_LAYERS - 2) // 2
CSA_NUM_LAYERS = FWD_NUM_LAYERS - 2 - HCA_NUM_LAYERS

# FP32 copy tile: 4 tokens x 1 HC lane x D = 64 KiB.
COPY_TOKEN_TILE = 4
MOE_ID_COPY_TILE = 8
assert ATTN_TILE_ROWS % COPY_TOKEN_TILE == 0
assert MOE_ROWS % MOE_ID_COPY_TILE == 0


@pl.jit.inline
def _fwd_attention_stage_barrier_from_completion(
    completion_token: pl.Tensor[[NUM_ATTN_TILES, 1, 8], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """Complete the attention stage before dispatching MoE."""
    stage_token = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_attn_stage_token", allow_early_resolve=False) as stage_tid:
        stage_token[0:1, 0:1, 0:8] = pl.slice(completion_token, [1, 1, 8], [0, 0, 0] )
    # Complete attention before constructing the dependent MoE graph.
    _completed = pl.read(stage_token, [0, 0, 0])
    return stage_tid


@pl.jit.inline
def _fwd_attention_stage_barrier_from_x_attn(
    x_attn: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    """HCA counterpart: one task samples every attention output tile."""
    x_attn_flat = pl.reshape(x_attn, [MOE_ROWS, HC_MULT, D])
    stage_tokens = pl.create_tensor([NUM_ATTN_TILES, 1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="fwd_attn_stage_token", allow_early_resolve=False) as stage_tid:
        for tile in pl.range(NUM_ATTN_TILES):
            row0 = tile * ATTN_TILE_ROWS
            stage_tokens[tile : tile + 1, 0:1, 0:8] = pl.slice(x_attn_flat, [1, 1, 8], [row0, 0, 0])
    # Complete attention before constructing the dependent MoE graph.
    _completed = pl.read(stage_tokens, [0, 0, 0])
    return stage_tid


@pl.jit.inline
def _fwd_pack_moe_inputs(
    x_attn: pl.Tensor[[MOE_ROWS, HC_MULT, D], pl.FP32],
    input_ids: pl.Tensor[[MOE_ROWS], pl.INT64],
    active_flat: pl.Tensor[[NUM_ATTN_TILES, OVERLAY_SOURCES], pl.INT32],
    packed_x: pl.Tensor[[MOE_ROWS, HC_MULT, D], pl.FP32],
    packed_ids: pl.Tensor[[MOE_ROWS], pl.INT64],
    attention_done_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.INT32]:
    """Pack the two real segment prefixes and zero the unused MoE capacity."""
    first_tokens = pl.cast(0, pl.INDEX)
    second_tokens = pl.cast(0, pl.INDEX)
    for tile in pl.range(MAX_SEGMENT_TILES):
        first_tokens = first_tokens + pl.cast(pl.read(active_flat, [tile, 1]), pl.INDEX)
        second_tokens = second_tokens + pl.cast(pl.read(active_flat, [MAX_SEGMENT_TILES + tile, 1]), pl.INDEX )
    active_tokens = first_tokens + second_tokens
    for block in pl.spmd(
        (MOE_ROWS // COPY_TOKEN_TILE) * HC_MULT,
        name_hint="pack_moe_hidden", deps=[attention_done_tid],
    ):
        row0 = (block // HC_MULT) * COPY_TOKEN_TILE
        hc_lane = block % HC_MULT
        for lane in pl.range(COPY_TOKEN_TILE):
            row = row0 + lane
            if row < active_tokens:
                source = row
                if row >= first_tokens:
                    source = MAX_SEGMENT_TILES * ATTN_TILE_ROWS + row - first_tokens
                packed_x[row : row + 1, hc_lane : hc_lane + 1, :] = pl.slice(x_attn, [1, 1, D], [source, hc_lane, 0] )
            else:
                packed_x[row : row + 1, hc_lane : hc_lane + 1, :] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0 )
    for block in pl.spmd(MOE_ROWS // MOE_ID_COPY_TILE, name_hint="pack_moe_ids"):
        for lane in pl.range(MOE_ID_COPY_TILE):
            row = block * MOE_ID_COPY_TILE + lane
            value = pl.cast(0, pl.INT64)
            if row < active_tokens:
                source = row
                if row >= first_tokens:
                    source = MAX_SEGMENT_TILES * ATTN_TILE_ROWS + row - first_tokens
                value = pl.read(input_ids, [source])
            pl.write(packed_ids, [row], value)
    return pl.cast(active_tokens, pl.INT32)


@pl.jit.inline
def _fwd_publish_hidden(
    x_next_work: pl.Tensor[[MOE_ROWS, HC_MULT, D], pl.FP32],
    active_flat: pl.Tensor[[NUM_ATTN_TILES, OVERLAY_SOURCES], pl.INT32],
    hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    moe_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Tensor[
    [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32
]:
    """Restore packed MoE rows to the two attention slots and zero padding."""
    hidden_flat = pl.reshape(hidden_out, [MOE_ROWS, HC_MULT, D])
    first_tokens = pl.cast(0, pl.INDEX)
    for first_tile in pl.range(MAX_SEGMENT_TILES):
        first_tokens = first_tokens + pl.cast(pl.read(active_flat, [first_tile, 1]), pl.INDEX)
    tile_blocks = (ATTN_TILE_ROWS // COPY_TOKEN_TILE) * HC_MULT
    with pl.spmd(NUM_ATTN_TILES * tile_blocks, name_hint="publish_hidden", deps=[moe_tid]) as _publish_tid:
        block = pl.tile.get_block_idx()
        tile = block // tile_blocks
        tile_block = block % tile_blocks
        token_block = tile_block // HC_MULT
        hc_lane = tile_block % HC_MULT
        token0 = token_block * COPY_TOKEN_TILE
        active = pl.read(active_flat, [tile, 1])
        for dt in pl.range(COPY_TOKEN_TILE):
            token = token0 + dt
            row = tile * ATTN_TILE_ROWS + token
            if token < active:
                packed_row = row
                if tile >= MAX_SEGMENT_TILES:
                    packed_row = first_tokens + row - MAX_SEGMENT_TILES * ATTN_TILE_ROWS
                hidden_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.slice(x_next_work, [1, 1, D], [packed_row, hc_lane, 0])
            else:
                hidden_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)
    return pl.reshape(hidden_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D])


@pl.jit.inline
def _fwd_wait_previous_moe(
    attention_tid: pl.Scalar[pl.TASK_ID],
    previous_completion: pl.Tensor[[1, 1, 8], pl.FP32],
) -> pl.Scalar[pl.TASK_ID]:
    # Empty CP ranks may not read the preceding hidden at all. Keep their
    # source-side communication epochs ordered by the explicit completion.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="fwd_previous_moe_complete",
        deps=[attention_tid],
        allow_early_resolve=False,
    ) as completed_tid:
        _previous = pl.read(previous_completion, [0, 0, 0])
    return completed_tid


@pl.jit.inline
def _fwd_moe_tail(
    x_attn: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D],
        pl.FP32,
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    input_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS], pl.INT64
    ],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    moe_x_mixed: pl.InOut[pl.Tensor[[MOE_ROWS, D], pl.BF16]],
    moe_post_ffn: pl.InOut[pl.Tensor[[MOE_ROWS, HC_MULT], pl.FP32]],
    moe_comb_ffn: pl.InOut[pl.Tensor[[MOE_ROWS, HC_MULT * HC_MULT], pl.FP32]],
    moe_ffn_out: pl.InOut[pl.Tensor[[MOE_ROWS, D], pl.BF16]],
    moe_dense_scale: pl.InOut[pl.Tensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], pl.FP32]],
    moe_returned_y: pl.InOut[pl.Tensor[[PREFILL_MOE_ROUTES_PER_SRC, D], pl.BF16]],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    prefill_moe_x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    prefill_moe_reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    prefill_moe_reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    completion_anchor: pl.Out[pl.Tensor[[1, 1, 8], pl.FP32]],
    attention_done_tid: pl.Scalar[pl.TASK_ID],
    layer_id: pl.Scalar[pl.INT32],
) -> pl.Tensor[
    [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], pl.FP32
]:
    """Pack real CP rows, run one MoE slab, and restore the attention slots."""
    x_attn_flat = pl.reshape(x_attn, [MOE_ROWS, HC_MULT, D])
    x_next_work = pl.create_tensor([MOE_ROWS, HC_MULT, D], dtype=pl.FP32)
    active_flat = pl.reshape(overlay_active_lengths, [NUM_ATTN_TILES, OVERLAY_SOURCES] )
    input_ids_flat = pl.reshape(input_ids, [MOE_ROWS])
    packed_x = pl.create_tensor([MOE_ROWS, HC_MULT, D], dtype=pl.FP32)
    packed_ids = pl.create_tensor([MOE_ROWS], dtype=pl.INT64)
    active_tokens = _fwd_pack_moe_inputs(
        x_attn_flat, input_ids_flat, active_flat, packed_x, packed_ids, attention_done_tid
    )

    moe_dense_x = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    moe_tid = prefill_moe(
        packed_x,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, packed_ids,
        routed_w1, routed_w1_scale,
        routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_next_work,
        moe_x_mixed, moe_post_ffn, moe_comb_ffn, moe_ffn_out,
        moe_dense_x, moe_dense_scale,

        moe_returned_y,
        count_target, count_signal,
        prefill_moe_x_target, prefill_moe_x_signal,
        prefill_moe_scale_target,
        prefill_moe_reverse_target, prefill_moe_reverse_signal,
        attention_done_tid, layer_id,
        pl.cast(layer_id + 1, pl.INT32),
        active_tokens,
    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="local1024_moe_completion_anchor",
        deps=[moe_tid],
        allow_early_resolve=False,
    ):
        final_element = pl.slice(x_next_work, [1, 1, 8], [MOE_ROWS - 1, 0, 0])
        completion_anchor[0:1, 0:1, 0:8] = final_element

    hidden_out = _fwd_publish_hidden(
        x_next_work, active_flat, hidden_out, moe_tid
    )
    return hidden_out


# ---------------------------------------------------------------------------
# Rank-local forward child
# ---------------------------------------------------------------------------
@pl.jit.inline
def _prefill_cp_metadata(
    control: pl.Tensor[[1, 16], pl.INT32],
    ori_block_table: pl.Tensor[[ORI_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_position_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_token_to_request: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT32],
    leaf_main_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_idx_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_main_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_inner_state_slots_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64],
    leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Build cold-request CP coordinates and physical slots on the owning device."""
    query_position_ids_flat = pl.reshape(query_position_ids, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), TAIL_ROWS] )
    query_token_to_request_flat = pl.reshape(query_token_to_request, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), TAIL_ROWS] )
    overlay_position_ids_flat = pl.reshape(overlay_position_ids, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_ROWS] )
    overlay_token_to_request_flat = pl.reshape(
        overlay_token_to_request, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_ROWS]
    )
    overlay_active_lengths_flat = pl.reshape(
        overlay_active_lengths, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES), OVERLAY_SOURCES]
    )
    swa_indices_flat = pl.reshape(swa_indices, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES) * (TAIL_ROWS), WIN] )
    cmp_indices_flat = pl.reshape(cmp_indices, [(LOCAL_PARTS) * (MAX_SEGMENT_TILES) * (TAIL_ROWS), IDX_TOPK] )
    leaf_positions_input_flat = pl.reshape(
        leaf_positions_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_main_slots_input_flat = pl.reshape(
        leaf_main_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_idx_slots_input_flat = pl.reshape(
        leaf_idx_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_main_state_slots_input_flat = pl.reshape(
        leaf_main_state_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )
    leaf_inner_state_slots_input_flat = pl.reshape(
        leaf_inner_state_slots_input, [(LOCAL_PARTS) * (CSA_MAX_COMPRESS_LEAVES), ATTN_TILE_ROWS]
    )

    for cp_request_segments_block in pl.spmd(1, name_hint="cp_request_segments"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        pl.write(cache_owner_rank_t, [0], pl.read(control, [0, 1]))
        pl.write(final_segment_t, [0], pl.cast((length - 1) // span, pl.INT32))
        for segment in pl.range(NUM_SEGMENTS):
            start = segment * span
            active = pl.max(0, pl.min(span, length - start))
            if segment < CP_SIZE:
                owner = segment
                part = pl.cast(0, pl.INDEX)
            else:
                owner = NUM_SEGMENTS - 1 - segment
                part = pl.cast(1, pl.INDEX)
            pl.write(segment_starts_t, [segment], pl.cast(start, pl.INT32))
            pl.write(segment_lengths_t, [segment], pl.cast(active, pl.INT32))
            pl.write(owner_rank_table, [segment], pl.cast(owner, pl.INT32))
            pl.write(owner_part_table, [segment], pl.cast(part, pl.INT32))
            pl.write(reverse_index, [segment], pl.cast(2 * owner + part, pl.INT32))
            for row in pl.range(TAIL_ROWS):
                position = pl.cast(-1, pl.INDEX)
                if row < pl.min(TAIL_ROWS, active):
                    position = pl.cast(start + pl.max(0, active - TAIL_ROWS) + row, pl.INDEX)
                pl.write(segment_tail_positions, [segment, row], pl.cast(position, pl.INT32))
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                segment = pl.cast(my_rank, pl.INDEX)
            else:
                segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = segment * span
            active = pl.max(0, pl.min(span, length - start))
            end = start + active
            valid = pl.cast(0, pl.INDEX)
            if active > 0:
                valid = pl.min(TAIL_ROWS, end)
            pl.write(owner_segments_t, [part], pl.cast(segment, pl.INT32))
            pl.write(predecessor_segments, [part], pl.cast(segment - 1, pl.INT32))
            pl.write(segment_active_lengths, [part], pl.cast(active, pl.INT32))
            pl.write(snapshot_valid, [part], pl.cast(valid, pl.INT32))
            for row in pl.range(TAIL_ROWS):
                position = pl.cast(-1, pl.INDEX)
                if row < valid:
                    position = pl.cast(end - valid + row, pl.INDEX)
                pl.write(snapshot_positions, [part, row], pl.cast(position, pl.INT32))
        for row in pl.range(TAIL_ROWS):
            position = length - TAIL_ROWS + row
            source = pl.cast(-1, pl.INT32)
            source_row = pl.cast(-1, pl.INT32)
            slot = pl.cast(-1, pl.INT32)
            if position >= 0:
                segment = position // span
                start = segment * span
                active = pl.max(0, pl.min(span, length - start))
                source = pl.cast(segment, pl.INT32)
                source_row = pl.cast(position - start - pl.max(0, active - TAIL_ROWS), pl.INT32)
                page = pl.read(ori_block_table, [position // BLOCK_ROWS])
                if page >= 0:
                    slot = pl.cast(page * BLOCK_ROWS + position % BLOCK_ROWS, pl.INT32)
            pl.write(final_win_seg_src, [row], source)
            pl.write(final_win_row_src, [row], source_row)
            pl.write(final_slot_mapping, [row], slot)

    # One block owns all small active-length fields to avoid cache-line sharing.
    for cp_query_coordinates_block in pl.spmd(1, name_hint="cp_query_coordinates"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                query_segment = pl.cast(my_rank, pl.INDEX)
            else:
                query_segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = query_segment * span
            seg_length = pl.max(0, pl.min(span, length - start))
            for tile in pl.range(MAX_SEGMENT_TILES):
                active = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - tile * ATTN_TILE_ROWS))
                tile_start = start + tile * ATTN_TILE_ROWS
                if tile > 0:
                    pred_start = start + (tile - 1) * ATTN_TILE_ROWS
                    pred_length = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - (tile - 1) * ATTN_TILE_ROWS))
                else:
                    if query_segment > 0:
                        pred_seg_start = (query_segment - 1) * span
                        pred_seg_length = pl.max(0, pl.min(span, length - pred_seg_start))
                        pred_length = pl.min(TAIL_ROWS, pred_seg_length)
                        pred_start = pred_seg_start + pl.max(0, pred_seg_length - TAIL_ROWS)
                    else:
                        pred_start = pl.cast(0, pl.INDEX)
                        pred_length = pl.cast(0, pl.INDEX)
                pl.write(overlay_active_lengths_flat, [part * MAX_SEGMENT_TILES + tile, 0], pl.cast(pred_length, pl.INT32))
                pl.write(overlay_active_lengths_flat, [part * MAX_SEGMENT_TILES + tile, 1], pl.cast(active, pl.INT32))
                for row in pl.range(ATTN_TILE_ROWS):
                    query = pl.cast(0, pl.INT32)
                    request = pl.cast(-1, pl.INT32)
                    current = pl.cast(-1, pl.INT32)
                    previous = pl.cast(-1, pl.INT32)
                    previous_request = pl.cast(-1, pl.INT32)
                    if row < active:
                        query = pl.cast(tile_start + row, pl.INT32)
                        request = pl.cast(0, pl.INT32)
                        current = query
                    if row < pred_length:
                        previous = pl.cast(pred_start + row, pl.INT32)
                        previous_request = pl.cast(0, pl.INT32)
                    pl.write(query_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, row], query)
                    pl.write(query_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, row], request)
                    pl.write(overlay_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, row], previous)
                    pl.write(overlay_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, row], previous_request)
                    pl.write(overlay_position_ids_flat, [part * MAX_SEGMENT_TILES + tile, ATTN_TILE_ROWS + row], current)
                    pl.write(overlay_token_to_request_flat, [part * MAX_SEGMENT_TILES + tile, ATTN_TILE_ROWS + row], request)


    for block in pl.spmd(LOCAL_PARTS * MAX_SEGMENT_TILES):
        part = block // MAX_SEGMENT_TILES
        tile = block % MAX_SEGMENT_TILES
        for row in pl.range(ATTN_TILE_ROWS):
            query = pl.read(query_position_ids_flat, [block, row])
            request = pl.read(query_token_to_request_flat, [block, row])
            active = pl.cast(pl.read(overlay_active_lengths_flat, [block, 1]), pl.INDEX)
            pred_length = pl.cast(pl.read(overlay_active_lengths_flat, [block, 0]), pl.INDEX)
            tile_start = pl.cast(pl.read(overlay_position_ids_flat, [block, ATTN_TILE_ROWS]), pl.INDEX)
            pred_start = pl.cast(pl.read(overlay_position_ids_flat, [block, 0]), pl.INDEX)
            for col in pl.range(WIN):
                key = query - WIN + 1 + col
                index = pl.cast(-1, pl.INT32)
                if request >= 0:
                    if key >= tile_start:
                        if key < tile_start + active:
                            index = pl.cast(OVERLAY_BASE + ATTN_TILE_ROWS + key - tile_start, pl.INT32)
                    else:
                        if key >= pred_start:
                            if key < pred_start + pred_length:
                                index = pl.cast(OVERLAY_BASE + key - pred_start, pl.INT32)
                pl.write(swa_indices_flat, [(part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS + row, col], index)
            for col in pl.range(IDX_TOPK):
                index = pl.cast(-1, pl.INT32)
                if request >= 0:
                    if col < (query + 1) // HCA_COMPRESS_RATIO:
                        index = pl.cast(col, pl.INT32)
                pl.write(cmp_indices_flat, [(part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS + row, col], index)

    for cp_csa_leaf_coordinates_block in pl.spmd(1, name_hint="cp_csa_leaf_coordinates"):
        length = pl.cast(pl.read(control, [0, 2]), pl.INDEX)
        span = pl.cast(pl.read(control, [0, 3]), pl.INDEX)
        for part in pl.range(LOCAL_PARTS):
            if part == 0:
                leaf_segment = pl.cast(my_rank, pl.INDEX)
            else:
                leaf_segment = pl.cast(NUM_SEGMENTS - 1 - my_rank, pl.INDEX)
            start = leaf_segment * span
            seg_length = pl.max(0, pl.min(span, length - start))
            seed = pl.cast(0, pl.INDEX)
            if leaf_segment > 0:
                if seg_length > 0:
                    predecessor_length = pl.min(TAIL_ROWS, pl.max(0, pl.min(span, length - (leaf_segment - 1) * span)))
                    seed = pl.min(start % CSA_COMPRESS_RATIO + CSA_COMPRESS_RATIO, predecessor_length)
            for leaf in pl.range(CSA_MAX_COMPRESS_LEAVES):
                if leaf == 0:
                    active = seed
                else:
                    active = pl.max(0, pl.min(ATTN_TILE_ROWS, seg_length - (leaf - 1) * ATTN_TILE_ROWS))
                pl.write(leaf_num_tokens_input, [part, leaf], pl.cast(active, pl.INT32))
                for row in pl.range(ATTN_TILE_ROWS):
                    position = pl.cast(0, pl.INDEX)
                    main_slot = pl.cast(-1, pl.INT64)
                    inner_slot = pl.cast(-1, pl.INT64)
                    cmp_slot = pl.cast(-1, pl.INT64)
                    idx_slot = pl.cast(-1, pl.INT64)
                    if row < active:
                        if leaf == 0:
                            position = start - seed + row
                        else:
                            position = start + (leaf - 1) * ATTN_TILE_ROWS + row
                        if position // CSA_MAIN_STATE_BLOCK_SIZE < CSA_MAIN_STATE_MAX_BLOCKS:
                            page = pl.read(csa_compress_state_block_table, [position // CSA_MAIN_STATE_BLOCK_SIZE])
                            if page >= 0:
                                main_slot = pl.cast(page * CSA_MAIN_STATE_BLOCK_SIZE + position % CSA_MAIN_STATE_BLOCK_SIZE, pl.INT64)
                        if position // CSA_INNER_STATE_BLOCK_SIZE < CSA_INNER_STATE_MAX_BLOCKS:
                            page = pl.read(csa_inner_compress_state_block_table, [position // CSA_INNER_STATE_BLOCK_SIZE])
                            if page >= 0:
                                inner_slot = pl.cast(page * CSA_INNER_STATE_BLOCK_SIZE + position % CSA_INNER_STATE_BLOCK_SIZE, pl.INT64)
                        if leaf > 0:
                            if (position + 1) % CSA_COMPRESS_RATIO == 0:
                                logical = (position + 1) // CSA_COMPRESS_RATIO - 1
                                if logical // CSA_CMP_STORAGE_BLOCK_SIZE < PREFILL_CMP_MAX_BLOCKS:
                                    # Local compressor output uses the compact cyclic scratch pool.
                                    scratch_page = logical // CSA_CMP_STORAGE_BLOCK_SIZE % PREFILL_CMP_BLOCK_NUM
                                    cmp_slot = pl.cast(scratch_page * CSA_CMP_STORAGE_BLOCK_SIZE + logical % CSA_CMP_STORAGE_BLOCK_SIZE, pl.INT64)
                                if logical // CSA_CMP_STORAGE_BLOCK_SIZE < IDX_CACHE_MAX_BLOCKS:
                                    page = pl.read(idx_block_table, [logical // CSA_CMP_STORAGE_BLOCK_SIZE])
                                    if page >= 0:
                                        idx_slot = pl.cast(page * CSA_CMP_STORAGE_BLOCK_SIZE + logical % CSA_CMP_STORAGE_BLOCK_SIZE, pl.INT64)
                    pl.write(leaf_positions_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], pl.cast(position, pl.INT32))
                    pl.write(leaf_main_state_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], main_slot)
                    pl.write(leaf_inner_state_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], inner_slot)
                    pl.write(leaf_main_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], cmp_slot)
                    pl.write(leaf_idx_slots_input_flat, [part * CSA_MAX_COMPRESS_LEAVES + leaf, row], idx_slot)
    return (
        segment_starts_t,
        predecessor_segments,
        query_position_ids,
        query_token_to_request,
        overlay_position_ids,
        overlay_token_to_request,
        overlay_active_lengths,
        swa_indices,
        reverse_index,
        owner_rank_table,
        final_win_seg_src,
        final_win_row_src,
        final_slot_mapping,
        segment_active_lengths,
        cache_owner_rank_t,
        owner_segments_t,
        final_segment_t,
        segment_tail_positions,
        snapshot_positions,
        snapshot_valid,
        owner_part_table,
        cmp_indices,
        segment_lengths_t,
        leaf_positions_input,
        leaf_main_slots_input,
        leaf_idx_slots_input,
        leaf_main_state_slots_input,
        leaf_inner_state_slots_input,
        leaf_num_tokens_input,
    )


@pl.jit.inline(auto_scope=False)
def prefill_cp_fwd(
    x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32],
    # SWA attention weights (layer-stacked: FWD_NUM_LAYERS * <unit>).
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # Raw KV pools use one caller-sized physical block span per layer.
    kv_cache: pl.InOut[
        pl.Tensor[
            [FWD_RAW_BLOCKS_DYN, BLOCK_ROWS, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    # Compressed KV pools are sliced by each attention type's layer ordinal.
    # One compressed-KV pool per flavour: a cache block holds
    # BLOCK_SIZE / COMPRESS_RATIO rows, which differs between HCA and CSA.
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [HCA_FWD_CMP_BLOCKS_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [CSA_FWD_CMP_BLOCKS_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_token_to_request: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_position_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_token_to_request: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    # --- Shared CP metadata needed by CSA/HCA cores (beyond SWA) ----------
    # Request ownership and segment metadata are shared across model layers.
    # CSA and HCA consume the relevant fields through their stage contracts.
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    # --- HCA type-specific (layers 3, 5, ..., 41) -----------------------------
    # HCA compact compressor weights (ratio-128: OUT_DIM == HEAD_DIM, so
    # cmp_wkv/cmp_wgate are [HEAD_DIM, D] and cmp_ape is [RATIO, HEAD_DIM]).
    # Stacked by HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L3, 1 for L5) per layer.
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # HCA persistent compressor state (rank-local InOut root; stacked by
    # HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...]).
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [HCA_FWD_STATE_BLOCKS_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    # HCA-specific metadata.
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    # --- CSA type-specific (layers 2, 4, ..., 42) -----------------------------
    # CSA main compressor weights (ratio-4: MAIN_OUT_DIM = 2*HEAD_DIM).
    # Stacked by CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L2, 1 for L4) per layer.
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[
        [CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32
    ],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # CSA indexer weights (stacked by CSA_NUM_LAYERS on axis 0).
    hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    # CSA inner compressor weights (ratio-4: INNER_OUT_DIM = 2*IDX_HEAD_DIM).
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[
        [CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32
    ],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    # CSA persistent state/caches (rank-local InOut roots; stacked by
    # CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...]).
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_FWD_STATE_BLOCKS_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                CSA_FWD_INNER_STATE_BLOCKS_DYN,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [CSA_FWD_IDX_BLOCKS_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[CSA_FWD_IDX_BLOCKS_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[[CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[
        [CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    # CSA-specific metadata.
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, ATTN_TILE_ROWS], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[[LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32],
    # Indexed by logical compressed page, whose row count is per-flavour.
    hca_cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    # --- Communication windows ------------------------------------------
    # Domain 1: shared tail exchange (SWA + CSA + HCA reuse one bank under
    # monotonic tail_comm_epoch). The dual-tail exchange also needs a
    # The legacy KV-tail window remains for CSA/HCA. Recipes-aligned SWA and
    # the compressor paths exchange normalized hidden tails through the
    # hidden-tail window, then project KV on the receiving rank.
    kv_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 2: HCA compact (one bank reused by HCA layers; compact_comm_epoch).
    cmp_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    cmp_meta_window: pld.DistributedTensor[[CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[[CP_SIZE, STATE_META_DIM], pl.INT32],
    hca_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hca_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 3: CSA compact/index/state (one bank reused by CSA layers).
    main_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], pl.BF16],
    idx_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8],
    scale_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP16],
    record_window: pld.DistributedTensor[[RECORDS_PER_WINDOW, META_DIM], pl.INT32],
    main_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, CSA_MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, CSA_INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    csa_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    csa_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # MoE weights (layer-stacked).
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS], pl.INT64],
    routed_w1: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    # Rank-local resident MoE workspaces, reused by every serialized layer.
    # Compact count/x/scale/reverse windows. all_to_all_v owns reusable,
    # self-clearing collective signals, so no per-wave epoch ABI remains.
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    prefill_moe_x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    prefill_moe_scale_target: pld.DistributedTensor[
        [PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32
    ],
    prefill_moe_reverse_target: pld.DistributedTensor[
        [PREFILL_MOE_TOTAL_CAP, D], pl.BF16
    ],
    prefill_moe_reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # Final normalization weights (HC head + final RMSNorm). The HC head
    # projects the [HC_MULT, D] hyper-connection mix to a single [D] row; the
    # final RMSNorm normalizes it into hidden_out for the LM head.
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    # Final outputs. pre_hc_hidden_out is the FP32 pre-HC MoE result (one
    # [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D] slab);
    # hidden_out is the BF16 post-RMSNorm final hidden ([LOCAL_ROWS, D]). The
    # host's next stage broadcasts its unique global-final row before LM head.
    # Both are pl.Out so the host ties them to host-level output slots.
    pre_hc_hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    hidden_out: pl.Out[pl.Tensor[[LOCAL_ROWS, D], pl.BF16]],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_ROWS, D], pl.BF16]:
    """Run the chronological attention/MoE schedule over caller-owned caches."""

    # Invocation-local MoE scratch is reused by the chronological layer loop.
    moe_x_mixed = pl.create_tensor([MOE_ROWS, D], dtype=pl.BF16)
    moe_post_ffn = pl.create_tensor([MOE_ROWS, HC_MULT], dtype=pl.FP32)
    moe_comb_ffn = pl.create_tensor([MOE_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    moe_ffn_out = pl.create_tensor([MOE_ROWS, D], dtype=pl.BF16)
    moe_dense_scale = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
    moe_returned_y = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.BF16)

    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    compressed_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = (
        pl.slice(freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0])
    )
    compressed_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = (
        pl.slice(freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0])
    )
    swa_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    swa_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        swa_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    compressed_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        compressed_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )
    compressed_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(
        compressed_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM]
    )

    raw_blocks = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_cmp_blocks = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    hca_state_blocks = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_cmp_blocks = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    csa_idx_blocks = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS
    csa_state_blocks = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    csa_inner_state_blocks = (
        pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS
    )
    # CSA scratch shares each sliced state root's physical page capacity.
    main_state_workspace0 = pl.create_tensor(
        [csa_state_blocks, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        dtype=pl.FP32,
    )
    main_state_workspace1 = pl.create_tensor(
        [csa_state_blocks, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        dtype=pl.FP32,
    )
    inner_state_workspace0 = pl.create_tensor(
        [csa_inner_state_blocks, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        dtype=pl.FP32,
    )
    inner_state_workspace1 = pl.create_tensor(
        [csa_inner_state_blocks, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        dtype=pl.FP32,
    )
    effective_x_workspace = pl.create_tensor([CSA_LOCAL_LEAVES * ATTN_TILE_ROWS, D], dtype=pl.BF16 )

    # Every layer uses the same local storage and monotonically increasing
    # protocol epochs. Empty ranks retain the explicit previous-MoE fence.
    layer_output = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], dtype=pl.FP32)
    moe_completion = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    x_attn = pl.create_tensor([LOCAL_PARTS, MAX_SEGMENT_TILES, ATTN_TILE_ROWS, HC_MULT, D], dtype=pl.FP32)
    # Carry the stage dependency beyond the attention scope, as in DSpark HCA.
    attention_stage_deps = pl.array.create(1, pl.TASK_ID)
    layer_hidden = x_hc
    for layer_id in pl.range(FWD_NUM_LAYERS):
        layer_index: pl.Scalar[pl.INT32] = pl.cast(layer_id, pl.INT32)
        kv_cache_layer = pl.slice(kv_cache, [raw_blocks, BLOCK_ROWS, 1, HEAD_DIM], [layer_index * raw_blocks, 0, 0, 0])
        hc_attn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0]
        )
        hc_attn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer_index * 3] )
        hc_attn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer_index * MIX_HC] )
        attn_norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [layer_index * D] )
        wq_a_layer: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [layer_index * D, 0] )
        wq_b_layer: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [layer_index * Q_LORA, 0]
        )
        wq_b_scale_layer: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [layer_index * H * HEAD_DIM]
        )
        wkv_layer: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [layer_index * D, 0] )
        gamma_cq_layer: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [layer_index * Q_LORA] )
        gamma_ckv_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [layer_index * HEAD_DIM] )
        attn_sink_layer: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [layer_index * H] )
        wo_a_layer: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
            wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [layer_index * O_GROUPS, 0, 0]
        )
        wo_b_layer: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(
            wo_b, [D, O_GROUPS * O_LORA], [layer_index * D, 0]
        )
        wo_b_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [layer_index * D] )
        hc_ffn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0]
        )
        hc_ffn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer_index * 3] )
        hc_ffn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer_index * MIX_HC] )
        norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [layer_index * D])
        gate_w_layer: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [layer_index * N_EXPERTS_GLOBAL, 0]
        )
        gate_bias_layer: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [layer_index * N_EXPERTS_GLOBAL]
        )
        tid2eid_layer: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [layer_index * VOCAB, 0] )
        routed_w1_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w1, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w1_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0]
        )
        routed_w3_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
            routed_w3, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w3_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0]
        )
        routed_w2_layer: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(
            routed_w2, [N_LOCAL, D, MOE_INTER], [layer_index * N_LOCAL, 0, 0]
        )
        routed_w2_scale_layer: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [layer_index * N_LOCAL, 0]
        )
        shared_w1_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [layer_index * MOE_INTER, 0]
        )
        shared_w1_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [layer_index * MOE_INTER]
        )
        shared_w3_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [layer_index * MOE_INTER, 0]
        )
        shared_w3_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [layer_index * MOE_INTER]
        )
        shared_w2_layer: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [layer_index * D, 0] )
        shared_w2_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [layer_index * D] )
        with pl.scope():
            attention_completion = pl.create_tensor([NUM_ATTN_TILES, 1, 8], dtype=pl.FP32)
            if layer_index < 2:
                prefill_cp_swa_core(
                    layer_hidden,
                    hc_attn_fn_layer,
                    hc_attn_scale_layer,
                    hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer,
                    wq_b_layer,
                    wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer,
                    gamma_ckv_layer,
                    swa_freqs_cos,
                    swa_freqs_sin,
                    kv_cache_layer,
                    attn_sink_layer,
                    wo_a_layer,
                    wo_b_layer,
                    wo_b_scale_layer,
                    segment_starts_t,
                    segment_tail_positions,
                    predecessor_segments,
                    query_position_ids,
                    query_token_to_request,
                    overlay_position_ids,
                    overlay_token_to_request,
                    overlay_active_lengths,
                    swa_indices,
                    reverse_index,
                    owner_rank_table,
                    final_win_seg_src,
                    final_win_row_src,
                    final_slot_mapping,
                    hidden_tail_window,
                    tail_ready,
                    tail_consumed,
                    x_attn,
                    attention_completion,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    layer_index,
                )
            elif layer_index % 2 == 0:
                type_index = (layer_index - 2) // 2
                cmp_kv_csa = pl.slice(
                    csa_cmp_kv,
                    [csa_cmp_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                    [type_index * csa_cmp_blocks, 0, 0, 0],
                )
                csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
                    csa_cmp_wkv,
                    [CSA_MAIN_OUT_DIM, D],
                    [type_index * CSA_MAIN_OUT_DIM, 0],
                )
                csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
                    csa_cmp_wgate,
                    [CSA_MAIN_OUT_DIM, D],
                    [type_index * CSA_MAIN_OUT_DIM, 0],
                )
                csa_cmp_ape_csa: pl.Tensor[
                    [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32
                ] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [type_index * CSA_COMPRESS_RATIO, 0])
                csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                    csa_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM]
                )
                hadamard_idx_csa: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = (
                    pl.slice(hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM, 0])
                )
                idx_wq_b_csa: pl.Tensor[
                    [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8
                ] = pl.slice(idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [type_index * Q_LORA, 0])
                idx_wq_b_scale_csa: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = (
                    pl.slice(idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [type_index * IDX_N_HEADS * IDX_HEAD_DIM])
                )
                idx_weights_proj_csa: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(
                    idx_weights_proj, [D, IDX_N_HEADS], [type_index * D, 0]
                )
                csa_inner_wkv_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = (
                    pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [type_index * CSA_INNER_OUT_DIM, 0])
                )
                csa_inner_wgate_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = (
                    pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [type_index * CSA_INNER_OUT_DIM, 0])
                )
                csa_inner_ape_csa: pl.Tensor[
                    [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32
                ] = pl.slice(
                    csa_inner_ape,
                    [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
                    [type_index * CSA_COMPRESS_RATIO, 0],
                )
                csa_inner_norm_w_csa: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(
                    csa_inner_norm_w, [IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM]
                )
                csa_compress_state_csa = pl.slice(
                    csa_compress_state,
                    [csa_state_blocks, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
                    [type_index * csa_state_blocks, 0, 0],
                )
                csa_inner_compress_state_csa = pl.slice(
                    csa_inner_compress_state,
                    [
                        csa_inner_state_blocks,
                        CSA_INNER_STATE_BLOCK_SIZE,
                        CSA_INNER_STATE_DIM,
                    ],
                    [type_index * csa_inner_state_blocks, 0, 0],
                )
                idx_kv_cache_csa = pl.slice(
                    idx_kv_cache,
                    [csa_idx_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
                    [type_index * csa_idx_blocks, 0, 0, 0],
                )
                idx_kv_scale_csa = pl.slice(
                    idx_kv_scale,
                    [csa_idx_blocks, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
                    [type_index * csa_idx_blocks, 0, 0, 0],
                )
                prefill_cp_csa_core(
                    layer_hidden,
                    hc_attn_fn_layer,
                    hc_attn_scale_layer,
                    hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer,
                    wq_b_layer,
                    wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer,
                    gamma_ckv_layer,
                    compressed_freqs_cos,
                    compressed_freqs_sin,
                    csa_cmp_wkv_csa,
                    csa_cmp_wgate_csa,
                    csa_cmp_ape_csa,
                    csa_cmp_norm_w_csa,
                    hadamard_idx_csa,
                    idx_wq_b_csa,
                    idx_wq_b_scale_csa,
                    idx_weights_proj_csa,
                    csa_inner_wkv_csa,
                    csa_inner_wgate_csa,
                    csa_inner_ape_csa,
                    csa_inner_norm_w_csa,
                    main_state_workspace0,
                    inner_state_workspace0,
                    main_state_workspace1,
                    inner_state_workspace1,
                    csa_compress_state_csa,
                    csa_compress_state_block_table,
                    csa_inner_compress_state_csa,
                    csa_inner_compress_state_block_table,
                    kv_cache_layer,
                    cmp_kv_csa,
                    csa_cmp_block_table,
                    idx_kv_cache_csa,
                    idx_kv_scale_csa,
                    idx_block_table,
                    segment_starts_t,
                    segment_lengths_t,
                    segment_active_lengths,
                    owner_segments_t,
                    predecessor_segments,
                    query_position_ids,
                    query_token_to_request,
                    overlay_position_ids,
                    overlay_token_to_request,
                    overlay_active_lengths,
                    swa_indices,
                    final_segment_t,
                    reverse_index,
                    owner_rank_table,
                    final_win_seg_src,
                    final_win_row_src,
                    final_slot_mapping,
                    leaf_positions_input,
                    leaf_main_slots_input,
                    leaf_idx_slots_input,
                    leaf_main_state_slots_input,
                    leaf_inner_state_slots_input,
                    leaf_num_tokens_input,
                    effective_x_workspace,
                    hidden_tail_window,
                    kv_tail_window,
                    tail_ready,
                    tail_consumed,
                    main_window,
                    idx_window,
                    scale_window,
                    record_window,
                    main_state_window,
                    main_state_meta_window,
                    inner_state_window,
                    inner_state_meta_window,
                    csa_compact_ready,
                    csa_compact_consumed,
                    attn_sink_layer,
                    wo_a_layer,
                    wo_b_layer,
                    wo_b_scale_layer,
                    x_attn,
                    attention_completion,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    layer_index,
                    type_index,
                )
            else:
                type_index = (layer_index - 3) // 2
                cmp_kv_hca = pl.slice(
                    hca_cmp_kv,
                    [hca_cmp_blocks, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                    [type_index * hca_cmp_blocks, 0, 0, 0],
                )
                hca_cmp_wkv_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(
                    hca_cmp_wkv, [HEAD_DIM, D], [type_index * HEAD_DIM, 0]
                )
                hca_cmp_wgate_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(
                    hca_cmp_wgate, [HEAD_DIM, D], [type_index * HEAD_DIM, 0]
                )
                hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32] = (
                    pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HEAD_DIM], [type_index * HCA_COMPRESS_RATIO, 0])
                )
                hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                    hca_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM]
                )
                hca_compress_state_hca = pl.slice(
                    hca_compress_state,
                    [hca_state_blocks, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
                    [type_index * hca_state_blocks, 0, 0],
                )
                prefill_cp_hca_core(
                    layer_hidden,
                    hc_attn_fn_layer,
                    hc_attn_scale_layer,
                    hc_attn_base_layer,
                    attn_norm_w_layer,
                    wq_a_layer,
                    wq_b_layer,
                    wq_b_scale_layer,
                    wkv_layer,
                    gamma_cq_layer,
                    gamma_ckv_layer,
                    compressed_freqs_cos,
                    compressed_freqs_sin,
                    hca_cmp_wkv_hca,
                    hca_cmp_wgate_hca,
                    hca_cmp_ape_hca,
                    hca_cmp_norm_w_hca,
                    hca_compress_state_hca,
                    hca_compress_state_block_table,
                    kv_cache_layer,
                    cmp_kv_hca,
                    hca_cmp_block_table,
                    segment_starts_t,
                    segment_active_lengths,
                    owner_segments_t,
                    predecessor_segments,
                    query_position_ids,
                    query_token_to_request,
                    overlay_position_ids,
                    overlay_token_to_request,
                    overlay_active_lengths,
                    swa_indices,
                    cmp_indices,
                    segment_tail_positions,
                    snapshot_positions,
                    snapshot_valid,
                    final_segment_t,
                    reverse_index,
                    owner_rank_table,
                    owner_part_table,
                    final_win_seg_src,
                    final_win_row_src,
                    final_slot_mapping,
                    hidden_tail_window,
                    kv_tail_window,
                    tail_ready,
                    tail_consumed,
                    cmp_window,
                    cmp_meta_window,
                    state_window,
                    state_meta_window,
                    hca_compact_ready,
                    hca_compact_consumed,
                    attn_sink_layer,
                    wo_a_layer,
                    wo_b_layer,
                    wo_b_scale_layer,
                    x_attn,
                    pl.read(cache_owner_rank_t, [0]),
                    my_rank,
                    layer_index,
                    type_index,
                )
            if layer_index < 2:
                attention_tid = _fwd_attention_stage_barrier_from_completion(attention_completion)
                attention_stage_deps[0] = attention_tid
            elif layer_index % 2 == 0:
                attention_tid = _fwd_attention_stage_barrier_from_completion(attention_completion)
                attention_stage_deps[0] = attention_tid
            else:
                attention_tid = _fwd_attention_stage_barrier_from_x_attn(x_attn)
                attention_stage_deps[0] = attention_tid
        attention_done = attention_stage_deps[0]
        if layer_index > 0:
            moe_ready = _fwd_wait_previous_moe(attention_done, moe_completion)
        else:
            moe_ready = attention_done
        with pl.scope():
            _fwd_moe_tail(
                x_attn,
                overlay_active_lengths,
                input_ids,
                hc_ffn_fn_layer,
                hc_ffn_scale_layer,
                hc_ffn_base_layer,
                norm_w_layer,
                gate_w_layer,
                gate_bias_layer,
                tid2eid_layer,
                routed_w1_layer,
                routed_w1_scale_layer,
                routed_w3_layer,
                routed_w3_scale_layer,
                routed_w2_layer,
                routed_w2_scale_layer,
                shared_w1_layer,
                shared_w1_scale_layer,
                shared_w3_layer,
                shared_w3_scale_layer,
                shared_w2_layer,
                shared_w2_scale_layer,
                moe_x_mixed,
                moe_post_ffn,
                moe_comb_ffn,
                moe_ffn_out,
                moe_dense_scale,
                moe_returned_y,
                count_target,
                count_signal,
                prefill_moe_x_target,
                prefill_moe_x_signal,
                prefill_moe_scale_target,
                prefill_moe_reverse_target,
                prefill_moe_reverse_signal,
                layer_output,
                moe_completion,
                moe_ready,
                layer_index,
            )
        layer_hidden = layer_output

    active_flat = pl.reshape(overlay_active_lengths, [NUM_ATTN_TILES, OVERLAY_SOURCES])
    pre_hc_hidden_out_flat = pl.reshape(pre_hc_hidden_out, [MOE_ROWS, HC_MULT, D])
    publish_src_flat = pl.reshape(layer_output, [MOE_ROWS, HC_MULT, D])
    publish_anchor = moe_completion

    # Retire communication credits, publish hidden states and apply HC/RMSNorm.
    with pl.scope():
        # Serving retains these windows without a host reset. Layer epochs
        # restart in each request, so retire all attention credits after the
        # final MoE and before the next HOST dispatch can reuse the windows.
        _clear_prefill_cp_exchange_signals(
            publish_anchor, tail_ready, tail_consumed,
            pl.cast(FWD_NUM_LAYERS, pl.INT32), my_rank,
        )
        _clear_prefill_cp_exchange_signals(
            publish_anchor, hca_compact_ready, hca_compact_consumed,
            pl.cast(HCA_NUM_LAYERS, pl.INT32), my_rank,
        )
        _clear_prefill_cp_exchange_signals(
            publish_anchor, csa_compact_ready, csa_compact_consumed,
            pl.cast(CSA_NUM_LAYERS, pl.INT32), my_rank,
        )
        clear_prefill_moe_signals(
            publish_anchor,
            count_signal,
            prefill_moe_x_signal,
            prefill_moe_reverse_signal,
        )

        tile_blocks = (ATTN_TILE_ROWS // COPY_TOKEN_TILE) * HC_MULT
        with pl.spmd(NUM_ATTN_TILES * tile_blocks, name_hint="publish_pre_hc_hidden_out"):
            block = pl.tile.get_block_idx()
            tile = block // tile_blocks
            tile_block = block % tile_blocks
            token_block = tile_block // HC_MULT
            hc_lane = tile_block % HC_MULT
            token0 = token_block * COPY_TOKEN_TILE
            active = pl.read(active_flat, [tile, 1])
            for dt in pl.range(COPY_TOKEN_TILE):
                token = token0 + dt
                row = tile * ATTN_TILE_ROWS + token
                if token < active:
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.slice(publish_src_flat, [1, 1, D], [row, hc_lane, 0])
                else:
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)

        # HC head + final RMSNorm: collapse the [HC_MULT, D] hyper-connection
        # mix to one [D] row and normalize into the BF16 hidden_out. The
        # pre_hc_hidden_out slab is MOE_ROWS == LOCAL_ROWS, so the
        # hc_head's T_DYN extent binds to LOCAL_ROWS. The intermediate
        # hidden_head is the hc_head BF16 output and the rms_norm input.
        pre_hc_view = pl.reshape(pre_hc_hidden_out_flat, [LOCAL_ROWS, HC_MULT, D])
        hidden_head = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
        with pl.scope():
            hc_head(pre_hc_view, hc_head_fn, hc_head_scale, hc_head_base, hidden_head)
            rms_norm(hidden_head, final_norm_w, hidden_out)
    return hidden_out

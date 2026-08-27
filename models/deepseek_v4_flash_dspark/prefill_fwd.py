# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 Flash DSpark 43-layer layer-major DSA-CP prefill backbone."""

import argparse
import os

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run_jit
from pypto.ir.distributed_compiled_program import DistributedConfig

from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    combine,
    dispatch,
)
from config import FLASH as MODEL_CONFIG
from prefill_swa import (
    build_cp_tensor_specs as build_swa_attention_tensor_specs,
    prefill_attention_swa_cp,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_cp_tensor_specs as build_hca_attention_tensor_specs,
    prefill_attention_hca_cp,
)
from prefill_csa import (
    BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    CSA_CMP_BLOCK_NUM,
    CSA_ORI_BLOCK_NUM,
    CSA_STATE_BLOCK_NUM,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_MAX_BLOCKS,
    H,
    HEAD_DIM,
    IDX_CACHE_BLOCK_NUM,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    START_POS,
    build_cp_tensor_specs as build_csa_attention_tensor_specs,
    prefill_attention_csa_cp,
)
from hc_post import T_TILE as HC_POST_T_TILE
from hc_pre import hc_pre
from gate import gate
from expert_shared import expert_shared
from expert_routed import expert_routed
from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
    prefill_cp_token_allgather_step,
)


# Dynamic shape variables.
FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
FWD_GROUP_TOKENS_DYN = pl.dynamic("PREFILL_FWD_GROUP_TOKENS_DYN")
FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# model config
MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
FWD_LAST_LAYER = FWD_NUM_LAYERS - 1

# Layer schedule: SWA 0-1, CSA even 2-42, and HCA odd 3-41.

# TP-sharded output-projection layout.
O_PROJ_LOCAL_GROUPS = O_GROUPS // TP_SIZE
O_PROJ_LOCAL_COLS = O_PROJ_LOCAL_GROUPS * O_LORA
O_PROJ_FULL_ROWS = O_GROUPS * O_LORA

# Full-weight projection scratch and communication windows.
O_PROJ_SCRATCH_GROUPS = O_GROUPS
O_PROJ_SCRATCH_RANK = O_LORA
O_PROJ_SCRATCH_INPUT = O_GROUP_IN
O_PROJ_SCRATCH_D = D
O_PROJ_SCRATCH_COLS = O_PROJ_FULL_ROWS
O_PROJ_WO_A_WINDOW_ROWS = O_PROJ_FULL_ROWS if TP_SIZE > 1 else 1
O_PROJ_WO_A_WINDOW_COLS = O_GROUP_IN if TP_SIZE > 1 else 1
O_PROJ_WO_B_WINDOW_ROWS = D if TP_SIZE > 1 else 1
O_PROJ_WO_B_WINDOW_COLS = O_PROJ_FULL_ROWS if TP_SIZE > 1 else 1

# tiling
O_PROJ_WEIGHT_COPY_TILE = 16

# runtime
PREFILL_RING_HEAP = (4 * 1024 * 1024 * 1024,) * 4

assert O_PROJ_FULL_ROWS % O_PROJ_WEIGHT_COPY_TILE == 0 and D % O_PROJ_WEIGHT_COPY_TILE == 0
assert MODEL_NUM_LAYERS == FWD_NUM_LAYERS, "DeepSeek-V4 Flash hidden layer count changed"

# FWD-layer stacked tensors, indexed by layer 0-42.
FWD_LAYER_STACKED_NAMES = [
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale", "hca_cmp_kv", "csa_cmp_kv",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
]
# MoE tensors flattened along the layer-major first axis.
MOE_LAYER_STACKED_NAMES = [
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
]
# CSA tensors indexed by CSA order 0-20.
CSA_LAYER_STACKED_NAMES = [
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
]
# HCA tensors indexed by HCA order 0-19.
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_cmp_kv",
]
# Per-rank tensors shared by every layer.
SHARED_NAMES = [
    "freqs_cos", "freqs_sin",
    "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "ori_slot_mapping_full", "position_ids_local", "position_ids_full", "input_ids",
    "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
    "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
    "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
]

# Mutable KV and compressor-state pools.
CACHE_NAMES = {
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
}

# Rank-sharded resident weights.
RESIDENT_WEIGHT_NAMES = frozenset(
    [
        n
        for n in (*FWD_LAYER_STACKED_NAMES, *CSA_LAYER_STACKED_NAMES, *HCA_LAYER_STACKED_NAMES)
        if n not in CACHE_NAMES
    ]
    + ["freqs_cos", "freqs_sin"]
)

# Resident attention tensors selected by layer inside each child.
ATTENTION_RESIDENT_NAMES = RESIDENT_WEIGHT_NAMES.difference(MOE_LAYER_STACKED_NAMES,)
ATTENTION_LAYER_STACKED_NAMES = ATTENTION_RESIDENT_NAMES.difference({"freqs_cos", "freqs_sin"})
FLATTENED_LAYER_STACKED_NAMES = frozenset(MOE_LAYER_STACKED_NAMES).union(ATTENTION_LAYER_STACKED_NAMES, CACHE_NAMES,)

# Rank-sharded resident caches.
RESIDENT_CACHE_NAMES = frozenset(CACHE_NAMES)

# Caches returned to the following decode invocation.
RESIDENT_CACHE_OUTPUT_NAMES = RESIDENT_CACHE_NAMES


@pl.jit.inline(auto_scope=False)
def gather_o_proj_full_weights(
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Gather one layer's resident TP shards into reusable full weights."""
    # Wait for every peer to consume the preceding window epoch.
    previous_epoch = weight_epoch - pl.const(1, pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_reuse_wait") as reuse_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_consumed, offsets=[source_tp, 0],
                    expected=previous_epoch, cmp=pld.WaitCmp.Ge,
                )

    wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_LOCAL_COLS, O_GROUP_IN])
    with pl.spmd(TP_SIZE, name_hint="o_proj_weight_push", deps=[reuse_wait_tid], allow_early_resolve=True) as push_tid:
        peer_tp = pl.tile.get_block_idx()
        peer = group_base + pl.cast(peer_tp, pl.INT32)
        pld.tensor.put(
            dst=wo_a_window, peer=peer, src=wo_a_local_flat,
            dst_offsets=[tp_rank * O_PROJ_LOCAL_COLS, 0], src_offsets=[0, 0],
            shape=[O_PROJ_LOCAL_COLS, O_GROUP_IN],
            chunk_rows=O_PROJ_WEIGHT_COPY_TILE, chunk_cols=O_GROUP_IN,
        )
        pld.tensor.put(
            dst=wo_b_window, peer=peer, src=wo_b_local,
            dst_offsets=[0, tp_rank * O_PROJ_LOCAL_COLS], src_offsets=[0, 0],
            shape=[D, O_PROJ_LOCAL_COLS],
            chunk_rows=O_PROJ_WEIGHT_COPY_TILE, chunk_cols=O_PROJ_LOCAL_COLS,
        )
        if peer_tp != tp_rank:
            pld.system.notify(target=weight_ready, peer=peer, offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,)

    # Register remote payload completion before readback.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_ready_wait") as ready_wait_tid:
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.defer_wait(
                    signal=weight_ready, offsets=[source_tp, 0],
                    expected=weight_epoch, cmp=pld.WaitCmp.Ge,
                )

    wo_a_full_flat = pl.reshape(wo_a_full, [O_PROJ_FULL_ROWS, O_GROUP_IN])
    with pl.spmd(
        O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_a_readback",
        deps=[push_tid, ready_wait_tid],
    ) as wo_a_readback_tid:
        order = pl.read(o_proj_order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
            tile = pl.load(
                wo_a_window, [row, 0],
                [O_PROJ_WEIGHT_COPY_TILE, O_GROUP_IN],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(tile, [row, 0], wo_a_full_flat)

    with pl.spmd(
        D // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_wo_b_readback",
        deps=[push_tid, ready_wait_tid],
    ) as wo_b_readback_tid:
        order = pl.read(o_proj_order_fence, [0])
        if order >= 0:
            row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
            tile = pl.load(
                wo_b_window, [row, 0],
                [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_FULL_ROWS],
                target_memory=pl.MemorySpace.Vec,
            )
            pl.store(tile, [row, 0], wo_b_full)

    # Publish window consumption after both readbacks.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="o_proj_weight_consumed",
        deps=[wo_a_readback_tid, wo_b_readback_tid],
    ) as weights_ready_tid:
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=weight_consumed, peer=group_base + peer_tp,
                    offsets=[tp_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd,
                )

    return weights_ready_tid


if TP_SIZE == 1:
    @pl.jit.inline(auto_scope=False)
    def gather_o_proj_full_weights(
        wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
        wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
        wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
        wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
        wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
        weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        o_proj_order_fence: pl.Tensor[[1], pl.INT32],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        weight_epoch: pl.Scalar[pl.INT32],
    ) -> pl.Scalar[pl.TASK_ID]:
        """Copy TP1's resident full weights into the common projection ABI."""
        wo_a_local_flat = pl.reshape(wo_a_local, [O_PROJ_FULL_ROWS, O_GROUP_IN])
        wo_a_full_flat = pl.reshape(wo_a_full, [O_PROJ_FULL_ROWS, O_GROUP_IN])
        with pl.spmd(
            O_PROJ_FULL_ROWS // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_tp1_wo_a_copy",
        ) as wo_a_copy_tid:
            order = pl.read(o_proj_order_fence, [0])
            if order >= 0:
                row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
                tile = pl.load(
                    wo_a_local_flat, [row, 0],
                    [O_PROJ_WEIGHT_COPY_TILE, O_GROUP_IN],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(tile, [row, 0], wo_a_full_flat)

        with pl.spmd(
            D // O_PROJ_WEIGHT_COPY_TILE, name_hint="o_proj_tp1_wo_b_copy",
        ) as wo_b_copy_tid:
            order = pl.read(o_proj_order_fence, [0])
            if order >= 0:
                row = pl.tile.get_block_idx() * O_PROJ_WEIGHT_COPY_TILE
                tile = pl.load(
                    wo_b_local, [row, 0],
                    [O_PROJ_WEIGHT_COPY_TILE, O_PROJ_FULL_ROWS],
                    target_memory=pl.MemorySpace.Vec,
                )
                pl.store(tile, [row, 0], wo_b_full)
        return pl.system.task_dummy(deps=[wo_a_copy_tid, wo_b_copy_tid])


@pl.jit.inline(auto_scope=False)
def _prefill_swa_attention(
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    attn_stage: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
) -> pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]:
    """Slice and run one full-state SWA DSA-CP attention layer."""
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    cache_start = layer_id * ori_block_num
    kv_cache_l: pl.Tensor[[ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(
        kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [cache_start, 0, 0, 0]
    )

    mix_start = layer_id * MIX_HC
    scale_start = layer_id * 3
    d_start = layer_id * D
    o_proj_group_start = layer_id * O_PROJ_LOCAL_GROUPS
    q_lora_start = layer_id * Q_LORA
    q_head_start = layer_id * H * HEAD_DIM
    head_start = layer_id * HEAD_DIM
    attn_head_start = layer_id * H
    hc_attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start, 0])
    hc_attn_scale_l: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [scale_start])
    hc_attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [mix_start])
    attn_norm_w_l: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [d_start])
    wq_a_l: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [d_start, 0])
    wq_b_l: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start, 0])
    wq_b_scale_l: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start])
    wkv_l: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [d_start, 0])
    gamma_cq_l: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [q_lora_start])
    gamma_ckv_l: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [head_start])
    swa_cos_profile = freqs_cos[0:1, :, :]
    swa_sin_profile = freqs_sin[0:1, :, :]
    freqs_cos_l = pl.reshape(swa_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    freqs_sin_l = pl.reshape(swa_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    attn_sink_l: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [attn_head_start])
    wo_a_l: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start, 0, 0]
    )
    wo_b_l: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8] = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start, 0])
    wo_b_scale_l: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [d_start])
    weight_epoch = layer_id + pl.const(1, pl.INT32)
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_l, wo_b_l,
        wo_a_full, wo_b_full,
        wo_a_window, wo_b_window,
        weight_ready, weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    with pl.scope():
        kv_cache_l, attn_stage, _gather_signal = prefill_attention_swa_cp(
            x_hc,
            hc_attn_fn_l, hc_attn_scale_l, hc_attn_base_l,
            attn_norm_w_l, wq_a_l, wq_b_l, wq_b_scale_l,
            wkv_l, gamma_cq_l, gamma_ckv_l,
            freqs_cos_l, freqs_sin_l,
            kv_cache_l, ori_block_table, ori_slot_mapping_full,
            position_ids_local, position_ids_full,
            attn_sink_l, wo_a_full, wo_b_full, wo_b_scale_l,
            attn_stage,
            gather_window, gather_signal, group_base, tp_rank,
            o_proj_weight_dep,
        )
    return attn_stage


@pl.jit.inline(auto_scope=False)
def _prefill_hca_attention(
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32,],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16,],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    attn_stage: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    attention_order: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
) -> pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]:
    """Slice and run one full-state HCA DSA-CP attention layer."""
    state_block_num = pl.tensor.dim(compress_state, 0) // HCA_NUM_LAYERS
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    cmp_block_num = pl.tensor.dim(cmp_kv, 0) // HCA_NUM_LAYERS
    state_start = attention_order * state_block_num
    cache_start = layer_id * ori_block_num
    cmp_start = attention_order * cmp_block_num
    compress_state_l: pl.Tensor[
        [state_block_num, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32
    ] = pl.slice(
        compress_state,
        [state_block_num, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
        [state_start, 0, 0],
    )
    kv_cache_l: pl.Tensor[[ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(
        kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [cache_start, 0, 0, 0]
    )
    cmp_kv_l: pl.Tensor[
        [cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ] = pl.slice(
        cmp_kv,
        [cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
        [cmp_start, 0, 0, 0],
    )

    mix_start = layer_id * MIX_HC
    scale_start = layer_id * 3
    d_start = layer_id * D
    o_proj_group_start = layer_id * O_PROJ_LOCAL_GROUPS
    q_lora_start = layer_id * Q_LORA
    q_head_start = layer_id * H * HEAD_DIM
    head_start = layer_id * HEAD_DIM
    attn_head_start = layer_id * H
    compress_start = attention_order * HCA_MAIN_OUT_DIM
    ape_start = attention_order * HCA_COMPRESS_RATIO
    norm_start = attention_order * HEAD_DIM
    hc_attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start, 0])
    hc_attn_scale_l: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [scale_start])
    hc_attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [mix_start])
    attn_norm_w_l: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [d_start])
    wq_a_l: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [d_start, 0])
    wq_b_l: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start, 0])
    wq_b_scale_l: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start])
    wkv_l: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [d_start, 0])
    gamma_cq_l: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [q_lora_start])
    gamma_ckv_l: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [head_start])
    compressed_cos_profile = freqs_cos[1:2, :, :]
    compressed_sin_profile = freqs_sin[1:2, :, :]
    freqs_cos_l = pl.reshape(compressed_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    freqs_sin_l = pl.reshape(compressed_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    cmp_wkv_l: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(cmp_wkv, [HCA_MAIN_OUT_DIM, D], [compress_start, 0])
    cmp_wgate_l: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
        cmp_wgate, [HCA_MAIN_OUT_DIM, D], [compress_start, 0]
    )
    cmp_ape_l: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(
        cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [ape_start, 0]
    )
    cmp_norm_w_l: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(cmp_norm_w, [HEAD_DIM], [norm_start])
    attn_sink_l: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [attn_head_start])
    wo_a_l: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start, 0, 0]
    )
    wo_b_l: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8] = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start, 0])
    wo_b_scale_l: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [d_start])
    weight_epoch = layer_id + pl.const(1, pl.INT32)
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_l, wo_b_l,
        wo_a_full, wo_b_full,
        wo_a_window, wo_b_window,
        weight_ready, weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    with pl.scope():
        attn_stage, _gather_signal = prefill_attention_hca_cp(
            x_hc,
            hc_attn_fn_l, hc_attn_scale_l, hc_attn_base_l,
            attn_norm_w_l, wq_a_l, wq_b_l, wq_b_scale_l,
            wkv_l, gamma_cq_l, gamma_ckv_l,
            freqs_cos_l, freqs_sin_l,
            cmp_wkv_l, cmp_wgate_l, cmp_ape_l, cmp_norm_w_l,
            compress_state_l, compress_state_block_table,
            kv_cache_l, ori_slot_mapping_full, ori_block_table,
            cmp_kv_l, cmp_block_table,
            position_ids_local, position_ids_full,
            cmp_slot_mapping_full, state_slot_mapping_full,
            attn_sink_l, wo_a_full, wo_b_full, wo_b_scale_l,
            attn_stage,
            gather_window, gather_signal, group_base, tp_rank,
            o_proj_weight_dep,
        )
    return attn_stage


@pl.jit.inline(auto_scope=False)
def _prefill_csa_attention(
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32,],
    compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[
        [FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
        pl.FP32,
    ],
    inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    cmp_kv: pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16,],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8,],
    idx_kv_scale: pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    attn_stage: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    attention_order: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
) -> pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]:
    """Slice and run one full-state CSA DSA-CP attention layer."""
    state_block_num = pl.tensor.dim(compress_state, 0) // CSA_NUM_LAYERS
    inner_state_block_num = pl.tensor.dim(inner_compress_state, 0) // CSA_NUM_LAYERS
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    cmp_block_num = pl.tensor.dim(cmp_kv, 0) // CSA_NUM_LAYERS
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS
    state_start = attention_order * state_block_num
    inner_state_start = attention_order * inner_state_block_num
    cache_start = layer_id * ori_block_num
    cmp_start = attention_order * cmp_block_num
    idx_start = attention_order * idx_block_num
    compress_state_l: pl.Tensor[
        [state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32
    ] = pl.slice(
        compress_state,
        [state_block_num, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
        [state_start, 0, 0],
    )
    inner_compress_state_l: pl.Tensor[
        [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32
    ] = pl.slice(
        inner_compress_state,
        [inner_state_block_num, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
        [inner_state_start, 0, 0],
    )
    kv_cache_l: pl.Tensor[[ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(
        kv_cache, [ori_block_num, BLOCK_SIZE, 1, HEAD_DIM], [cache_start, 0, 0, 0]
    )
    cmp_kv_l: pl.Tensor[
        [cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
    ] = pl.slice(
        cmp_kv,
        [cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
        [cmp_start, 0, 0, 0],
    )
    idx_kv_cache_l: pl.Tensor[
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
    ] = pl.slice(
        idx_kv_cache,
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
        [idx_start, 0, 0, 0],
    )
    idx_kv_scale_l: pl.Tensor[
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32
    ] = pl.slice(
        idx_kv_scale,
        [idx_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1],
        [idx_start, 0, 0, 0],
    )

    mix_start = layer_id * MIX_HC
    scale_start = layer_id * 3
    d_start = layer_id * D
    o_proj_group_start = layer_id * O_PROJ_LOCAL_GROUPS
    q_lora_start = layer_id * Q_LORA
    q_head_start = layer_id * H * HEAD_DIM
    head_start = layer_id * HEAD_DIM
    attn_head_start = layer_id * H
    compress_start = attention_order * CSA_MAIN_OUT_DIM
    ape_start = attention_order * CSA_COMPRESS_RATIO
    norm_start = attention_order * HEAD_DIM
    idx_head_start = attention_order * IDX_HEAD_DIM
    idx_q_lora_start = attention_order * Q_LORA
    idx_q_head_start = attention_order * IDX_N_HEADS * IDX_HEAD_DIM
    idx_proj_start = attention_order * D
    inner_start = attention_order * INNER_OUT_DIM
    inner_norm_start = attention_order * IDX_HEAD_DIM
    hc_attn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [mix_start, 0])
    hc_attn_scale_l: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [scale_start])
    hc_attn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [mix_start])
    attn_norm_w_l: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [d_start])
    wq_a_l: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [d_start, 0])
    wq_b_l: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [q_lora_start, 0])
    wq_b_scale_l: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [q_head_start])
    wkv_l: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [d_start, 0])
    gamma_cq_l: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [q_lora_start])
    gamma_ckv_l: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [head_start])
    compressed_cos_profile = freqs_cos[1:2, :, :]
    compressed_sin_profile = freqs_sin[1:2, :, :]
    freqs_cos_l = pl.reshape(compressed_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    freqs_sin_l = pl.reshape(compressed_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    cmp_wkv_l: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(cmp_wkv, [CSA_MAIN_OUT_DIM, D], [compress_start, 0])
    cmp_wgate_l: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(
        cmp_wgate, [CSA_MAIN_OUT_DIM, D], [compress_start, 0]
    )
    cmp_ape_l: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(
        cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [ape_start, 0]
    )
    cmp_norm_w_l: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(cmp_norm_w, [HEAD_DIM], [norm_start])
    hadamard_idx_l: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(
        hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [idx_head_start, 0]
    )
    idx_wq_b_l: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(
        idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [idx_q_lora_start, 0]
    )
    idx_wq_b_scale_l: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(
        idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [idx_q_head_start]
    )
    idx_weights_proj_l: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(
        idx_weights_proj, [D, IDX_N_HEADS], [idx_proj_start, 0]
    )
    inner_wkv_l: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(inner_wkv, [INNER_OUT_DIM, D], [inner_start, 0])
    inner_wgate_l: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(inner_wgate, [INNER_OUT_DIM, D], [inner_start, 0])
    inner_ape_l: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32] = pl.slice(
        inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [ape_start, 0]
    )
    inner_norm_w_l: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(inner_norm_w, [IDX_HEAD_DIM], [inner_norm_start])
    attn_sink_l: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [attn_head_start])
    wo_a_l: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(
        wo_a, [O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], [o_proj_group_start, 0, 0]
    )
    wo_b_l: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8] = pl.slice(wo_b, [D, O_PROJ_LOCAL_COLS], [d_start, 0])
    wo_b_scale_l: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [d_start])
    weight_epoch = layer_id + pl.const(1, pl.INT32)
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_l, wo_b_l,
        wo_a_full, wo_b_full,
        wo_a_window, wo_b_window,
        weight_ready, weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    with pl.scope():
        attn_stage, _gather_signal = prefill_attention_csa_cp(
            x_hc,
            hc_attn_fn_l, hc_attn_scale_l, hc_attn_base_l,
            attn_norm_w_l, wq_a_l, wq_b_l, wq_b_scale_l,
            wkv_l, gamma_cq_l, gamma_ckv_l,
            freqs_cos_l, freqs_sin_l,
            cmp_wkv_l, cmp_wgate_l, cmp_ape_l, cmp_norm_w_l,
            compress_state_l, compress_state_block_table,
            hadamard_idx_l, idx_wq_b_l, idx_wq_b_scale_l, idx_weights_proj_l,
            inner_wkv_l, inner_wgate_l, inner_ape_l, inner_norm_w_l,
            inner_compress_state_l, inner_compress_state_block_table,
            kv_cache_l, ori_block_table, ori_slot_mapping_full,
            cmp_kv_l, cmp_block_table,
            idx_kv_cache_l, idx_kv_scale_l, idx_block_table,
            position_ids_local, position_ids_full,
            cmp_slot_mapping_full, idx_slot_mapping_full,
            state_slot_mapping_full, inner_state_slot_mapping_full,
            attn_sink_l, wo_a_full, wo_b_full, wo_b_scale_l,
            attn_stage,
            gather_window, gather_signal, group_base, tp_rank,
            o_proj_weight_dep,
        )
    return attn_stage


@pl.jit.inline
def _clear_moe_forward_signals(
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    completion_tid: pl.Scalar[pl.TASK_ID],
):
    """Clear all shared epochs after the final full-forward wave retires."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="moe_forward_signal_clear",
        deps=[completion_tid],
    ) as clear_tid:
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)
            pl.write(stage_done, [src, 0], zero)
    return clear_tid


@pl.jit.inline
def _complete_moe_wave(
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
    completion_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Publish one globally complete MoE wave before window reuse."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="moe_wave_done_notify",
        deps=[completion_tid],
    ) as notify_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.notify(
                    target=stage_done, peer=peer, offsets=[my_rank, 0],
                    value=1, op=pld.NotifyOp.AtomicAdd,
                )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="moe_wave_done_wait",
    ) as wait_tid:
        for peer in pl.range(N_RANKS):
            if peer != my_rank:
                pld.system.defer_wait(signal=stage_done, offsets=[peer, 0], expected=moe_epoch, cmp=pld.WaitCmp.Ge,)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="moe_wave_done_publish",
        deps=[completion_tid, notify_tid, wait_tid],
    ) as barrier_tid:
        pl.write(stage_token, [0], pl.cast(moe_epoch, pl.INT32))
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def _retire_o_proj_weight_signals(
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Retire one full forward's o-projection collective credits."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="o_proj_weight_signal_retire",
    ) as retire_tid:
        completed_layers = pl.read(o_proj_order_fence, [0])
        if completed_layers >= FWD_NUM_LAYERS:
            neg_epochs = pl.cast(0 - FWD_NUM_LAYERS, pl.INT32)
            self_rank = group_base + tp_rank
            for source_tp in pl.range(TP_SIZE):
                if source_tp != tp_rank:
                    pld.system.notify(
                        target=weight_ready, peer=self_rank,
                        offsets=[source_tp, 0], value=neg_epochs,
                        op=pld.NotifyOp.AtomicAdd,
                    )
                    pld.system.notify(
                        target=weight_consumed, peer=self_rank,
                        offsets=[source_tp, 0], value=neg_epochs,
                        op=pld.NotifyOp.AtomicAdd,
                    )
    return retire_tid


@pl.jit.inline
def _hc_post_after_token_allgather(
    x: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    residual: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    completion_tid: pl.Scalar[pl.TASK_ID],
):
    """Run full-state hc_post after one TP activation gather retires."""
    t_dim = pl.tensor.dim(x, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])
    token_tiles = (t_dim + HC_POST_T_TILE - 1) // HC_POST_T_TILE
    with pl.spmd(
        token_tiles * HC_MULT, name_hint="prefill_hc_post",
        deps=[completion_tid],
    ) as _hc_post_tid:
        token_block = pl.tile.get_block_idx() // HC_MULT
        out_h = pl.tile.get_block_idx() % HC_MULT
        t0 = token_block * HC_POST_T_TILE
        for token in pl.pipeline(t0, t0 + HC_POST_T_TILE, stage=2):
            if token < t_dim:
                post_w = pl.read(post, [token, out_h])
                x_row = pl.cast(x[token : token + 1, :], target_type=pl.FP32)
                y_row = pl.mul(x_row, post_w)
                for in_h in pl.pipeline(HC_MULT, stage=4):
                    comb_w = pl.read(comb, [token, in_h * HC_MULT + out_h])
                    res_d = in_h * D
                    residual_row = residual_flat[token : token + 1, res_d : res_d + D]
                    residual_weighted = pl.mul(residual_row, comb_w)
                    y_row = pl.add(y_row, residual_weighted)
                out_d = out_h * D
                y_flat[token : token + 1, out_d : out_d + D] = y_row
    return y


@pl.jit.inline(auto_scope=False)
def prefill_moe_post(
    ffn_out_local: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    residual_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    post_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_hc_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Restore the replicated layer state after token-SP MoE."""
    group_rows = pl.tensor.dim(residual_full, 0)
    ffn_out_full = pl.create_tensor([group_rows, D], dtype=pl.BF16)
    ffn_out_full, _gather_signal, gather_completion_tid = prefill_cp_token_allgather_step(
        ffn_out_local, ffn_out_full,
        gather_window, gather_signal,
        group_base, tp_rank,
    )
    x_hc_full = _hc_post_after_token_allgather(
        ffn_out_full, residual_full,
        post_full, comb_full,
        x_hc_full,
        gather_completion_tid,
    )
    return x_hc_full


@pl.jit.inline(auto_scope=False)
def prefill_moe_wave(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
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
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one rank-local MoE wave and publish its global completion."""
    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, TOPK], dtype=pl.INT32)
    weights = pl.create_tensor([T, TOPK], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, pl.const(T, pl.INT32), tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    shared_out = pl.create_tensor([T, D], dtype=pl.BF16)
    expert_shared(
        x_norm_i8, x_norm_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        shared_out,
    )

    recv_x_out = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.INT8)
    recv_scale_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_w_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.FP32, manual_dep=True)
    recv_r_route_out = pl.create_tensor([N_LOCAL, RECV_MAX], dtype=pl.INT32, manual_dep=True)
    recv_count_out = pl.create_tensor([N_LOCAL, 1], dtype=pl.INT32)
    recv_meta_local = pl.create_tensor([N_RANKS, N_LOCAL], dtype=pl.INT32, manual_dep=True)
    dispatch(
        indices, x_norm_i8, x_norm_scale, weights,
        recv_x_out, recv_scale_out, recv_w_out, recv_r_route_out,
        recv_count_out, recv_meta_local,
        recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
        pl.const(T, pl.INT32), my_rank, moe_epoch,
    )

    recv_y = pl.create_tensor([N_LOCAL, RECV_MAX, D], dtype=pl.BF16)
    expert_routed(
        recv_x_out, recv_scale_out, recv_w_out, recv_count_out,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    combine_tid = combine(
        recv_y, recv_r_route_out, shared_out,
        ffn_out, recv_meta_local,
        routed_y_buf, combine_arrived,
        pl.const(T, pl.INT32), my_rank, moe_epoch,
    )
    barrier_tid = _complete_moe_wave(stage_done, stage_token, my_rank, moe_epoch, combine_tid)
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def _run_prefill_moe_wave(
    x_mixed_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
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
    ffn_out: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    wave_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Run one fixed rank-local MoE wave from a replicated TP-group input."""
    local_rows = pl.tensor.dim(ffn_out, 0)
    local_wave_base = wave_id * T
    tp_rank = my_rank % TP_SIZE
    full_wave_base = tp_rank * local_rows + local_wave_base

    x_mixed_wave = pl.create_tensor([T, D], dtype=pl.BF16)
    for token in pl.spmd(T, name_hint="prefill_moe_x_mixed_stage"):
        wave_order = pl.read(stage_token, [0])
        if wave_order >= 0:
            full_token = full_wave_base + token
            x_mixed_wave[token : token + 1, :] = x_mixed_full[full_token : full_token + 1, :]
    input_id_count = pl.tensor.dim(input_ids, 0)
    input_ids_rows = pl.reshape(input_ids, [input_id_count, 1])
    input_ids_wave_rows = pl.create_tensor([T, 1], dtype=pl.INT64)
    for token_block in pl.spmd(T // 4, name_hint="prefill_moe_ids_stage"):
        token0 = token_block * 4
        local_token = local_wave_base + token0
        input_ids_wave_rows[token0 : token0 + 4, 0:1] = input_ids_rows[local_token : local_token + 4, 0:1]
    input_ids_wave = pl.reshape(input_ids_wave_rows, [T])

    d_start = layer_id * D
    expert_start = layer_id * N_EXPERTS_GLOBAL
    vocab_start = layer_id * VOCAB
    local_expert_start = layer_id * N_LOCAL
    moe_start = layer_id * MOE_INTER
    norm_w_l: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [d_start])
    gate_w_l: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [expert_start, 0])
    gate_bias_l: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [expert_start])
    tid2eid_l: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [vocab_start, 0])
    routed_w1_l: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w1, [N_LOCAL, MOE_INTER, D], [local_expert_start, 0, 0]
    )
    routed_w1_scale_l: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
        routed_w1_scale, [N_LOCAL, MOE_INTER], [local_expert_start, 0]
    )
    routed_w3_l: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(
        routed_w3, [N_LOCAL, MOE_INTER, D], [local_expert_start, 0, 0]
    )
    routed_w3_scale_l: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(
        routed_w3_scale, [N_LOCAL, MOE_INTER], [local_expert_start, 0]
    )
    routed_w2_l: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(
        routed_w2, [N_LOCAL, D, MOE_INTER], [local_expert_start, 0, 0]
    )
    routed_w2_scale_l: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
        routed_w2_scale, [N_LOCAL, D], [local_expert_start, 0]
    )
    shared_w1_l: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [moe_start, 0])
    shared_w1_scale_l: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [moe_start])
    shared_w3_l: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [moe_start, 0])
    shared_w3_scale_l: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [moe_start])
    shared_w2_l: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [d_start, 0])
    shared_w2_scale_l: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [d_start])

    ffn_wave = pl.create_tensor([T, D], dtype=pl.BF16)
    barrier_tid = prefill_moe_wave(
        x_mixed_wave,
        norm_w_l, gate_w_l, gate_bias_l, tid2eid_l, input_ids_wave,
        routed_w1_l, routed_w1_scale_l, routed_w3_l, routed_w3_scale_l,
        routed_w2_l, routed_w2_scale_l,
        shared_w1_l, shared_w1_scale_l, shared_w3_l, shared_w3_scale_l,
        shared_w2_l, shared_w2_scale_l,
        ffn_wave,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        stage_done, stage_token,
        layer_id, my_rank, moe_epoch,
    )
    ffn_out[local_wave_base : local_wave_base + T, :] = ffn_wave
    return barrier_tid


@pl.jit.inline(auto_scope=False)
def _prefill_moe(
    attn_out: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
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
    x_mixed: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_token: pl.Tensor[[1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]:
    """Run one layer's CP-aware MoE from full attention state to full hidden state."""
    mix_start = layer_id * MIX_HC
    scale_start = layer_id * 3
    hc_ffn_fn_l: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [mix_start, 0])
    hc_ffn_scale_l: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [scale_start])
    hc_ffn_base_l: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [mix_start])
    with pl.scope():
        hc_pre(attn_out, hc_ffn_fn_l, hc_ffn_scale_l, hc_ffn_base_l, x_mixed, post_ffn, comb_ffn,)

    num_tiles = pl.tensor.dim(ffn_out, 0) // T
    for wave in pl.range(num_tiles):
        wave_i32 = pl.cast(wave, pl.INT32)
        moe_epoch = layer_id * num_tiles + wave_i32 + pl.const(1, pl.INT32)
        is_last_layer = pl.cast(layer_id == FWD_LAST_LAYER, pl.INT32)
        is_last_wave = pl.cast(wave == num_tiles - 1, pl.INT32)
        finalize_moe = is_last_layer * is_last_wave
        with pl.scope():
            barrier_tid = _run_prefill_moe_wave(
                x_mixed,
                norm_w, gate_w, gate_bias, tid2eid, input_ids,
                routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
                routed_w2, routed_w2_scale,
                shared_w1, shared_w1_scale,
                shared_w3, shared_w3_scale,
                shared_w2, shared_w2_scale,
                ffn_out,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                stage_done, stage_token,
                layer_id, wave_i32, my_rank, moe_epoch,
            )
            if finalize_moe != 0:
                _clear_moe_forward_signals(arrived, data_arrived, combine_arrived, stage_done, barrier_tid,)

    with pl.scope():
        x_hc = prefill_moe_post(
            ffn_out, attn_out, post_ffn, comb_ffn, x_hc,
            gather_window, gather_signal, group_base, tp_rank,
        )
    return x_hc


@pl.jit(auto_scope=False)
def prefill_fwd(
    x_hc: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
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
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16,]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16,]],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE,
             HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE,
             CSA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE,
             CSA_INNER_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8,]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
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
    o_proj_wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    o_proj_wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    attn_stage: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run all 43 layers with a compact runtime SWA/CSA/HCA dispatch loop.

    The loop keeps one protocol body for lower program size; layer indices and
    stacked-tensor slices remain runtime values.
    """
    group_base = my_rank // TP_SIZE * TP_SIZE
    tp_rank = my_rank % TP_SIZE
    num_tiles = pl.tensor.dim(ffn_out, 0) // T
    stage_token = pl.create_tensor([1], dtype=pl.INT32)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_fwd_order_state_init"):
        pl.write(stage_token, [0], pl.cast(0, pl.INT32))
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))

    for layer in pl.range(FWD_NUM_LAYERS):
        layer_i32 = pl.cast(layer, pl.INT32)
        if layer < 2:
            _prefill_swa_attention(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                position_ids_local, position_ids_full,
                attn_sink, wo_a, wo_b, wo_b_scale,
                o_proj_wo_a_full, o_proj_wo_b_full,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank, layer_i32,
            )
        elif layer % 2 == 0:
            csa_order_i32 = pl.cast((layer - 2) // 2, pl.INT32)
            _prefill_csa_attention(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                csa_cmp_wkv, csa_cmp_wgate,
                csa_cmp_ape, csa_cmp_norm_w,
                csa_compress_state, csa_compress_state_block_table,
                csa_hadamard_idx, csa_idx_wq_b,
                csa_idx_wq_b_scale, csa_weights_proj,
                csa_inner_wkv, csa_inner_wgate,
                csa_inner_ape, csa_inner_norm_w,
                csa_inner_compress_state,
                csa_inner_compress_state_block_table,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                csa_cmp_kv, csa_cmp_block_table,
                idx_kv_cache, idx_kv_scale,
                idx_block_table, position_ids_local, position_ids_full,
                csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full,
                csa_state_slot_mapping_full,
                csa_inner_state_slot_mapping_full,
                attn_sink, wo_a, wo_b, wo_b_scale,
                o_proj_wo_a_full, o_proj_wo_b_full,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank, csa_order_i32, layer_i32,
            )
        else:
            hca_order_i32 = pl.cast((layer - 3) // 2, pl.INT32)
            _prefill_hca_attention(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                hca_cmp_wkv, hca_cmp_wgate,
                hca_cmp_ape, hca_cmp_norm_w,
                hca_compress_state, hca_compress_state_block_table,
                kv_cache, ori_slot_mapping_full, ori_block_table,
                hca_cmp_kv, hca_cmp_block_table,
                position_ids_local, position_ids_full,
                hca_cmp_slot_mapping_full, hca_state_slot_mapping_full,
                attn_sink, wo_a, wo_b, wo_b_scale,
                o_proj_wo_a_full, o_proj_wo_b_full,
                o_proj_wo_a_window, o_proj_wo_b_window,
                o_proj_weight_ready, o_proj_weight_consumed,
                o_proj_order_fence,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank, hca_order_i32, layer_i32,
            )

        _prefill_moe(
            attn_stage, x_hc,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale,
            shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            x_mixed, post_ffn, comb_ffn, ffn_out,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            stage_done, stage_token,
            gather_window, gather_signal,
            group_base, tp_rank, layer_i32, my_rank,
        )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="o_proj_order_commit",
        ):
            completed_wave = pl.read(stage_token, [0])
            expected_wave = (layer_i32 + pl.const(1, pl.INT32)) * num_tiles
            if completed_wave >= expected_wave:
                pl.write(o_proj_order_fence, [0], layer_i32 + pl.const(1, pl.INT32),)

    with pl.scope():
        _retire_o_proj_weight_signals(
            o_proj_order_fence,
            o_proj_weight_ready, o_proj_weight_consumed,
            group_base, tp_rank,
        )
    return x_hc


# DSA-CP layer-major multi-wave forward.
@pl.jit.host
def l3_prefill_fwd(
    x_hc: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16,]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16,],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_HCA_CMP_BLOCK_NUM_DYN,
             HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_CSA_CMP_BLOCK_NUM_DYN,
             CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN,
             HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN,
             CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN,
             INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_IDX_BLOCK_NUM_DYN,
             CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32,]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    o_proj_wo_a_full: pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16,],
    o_proj_wo_b_full: pl.Tensor[[N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    attn_stage: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16],
    post_ffn: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_ffn: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    ffn_out: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16],
):
    """Run layer-major DSA-CP with replicated boundaries and token-local compute."""
    x_hc.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    attn_stage.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ffn_out.bind_dynamic(1, FWD_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, FWD_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    stage_done_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        stage_done = pld.window(stage_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf,
            [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS],
            dtype=pl.BF16,
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf,
            [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS],
            dtype=pl.INT8,
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_fwd(
            x_hc[r],
            hc_attn_fn[r], hc_attn_scale[r], hc_attn_base[r],
            attn_norm_w[r], wq_a[r], wq_b[r], wq_b_scale[r],
            wkv[r], gamma_cq[r], gamma_ckv[r],
            kv_cache[r], attn_sink[r], wo_a[r], wo_b[r], wo_b_scale[r],
            hca_cmp_kv[r], csa_cmp_kv[r],
            hca_cmp_wkv[r], hca_cmp_wgate[r], hca_cmp_ape[r],
            hca_cmp_norm_w[r], hca_compress_state[r],
            csa_cmp_wkv[r], csa_cmp_wgate[r], csa_cmp_ape[r],
            csa_cmp_norm_w[r], csa_compress_state[r],
            csa_hadamard_idx[r], csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r], csa_weights_proj[r],
            csa_inner_wkv[r], csa_inner_wgate[r],
            csa_inner_ape[r], csa_inner_norm_w[r],
            csa_inner_compress_state[r], idx_kv_cache[r], idx_kv_scale[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            freqs_cos[r], freqs_sin[r],
            ori_block_table[r], hca_cmp_block_table[r],
            csa_cmp_block_table[r], idx_block_table[r],
            ori_slot_mapping_full[r], position_ids_local[r],
            position_ids_full[r], input_ids[r],
            hca_cmp_slot_mapping_full[r], hca_state_slot_mapping_full[r],
            csa_cmp_slot_mapping_full[r], csa_idx_slot_mapping_full[r],
            csa_state_slot_mapping_full[r], csa_inner_state_slot_mapping_full[r],
            hc_ffn_fn[r], hc_ffn_scale[r], hc_ffn_base[r],
            norm_w[r], gate_w[r], gate_bias[r], tid2eid[r],
            routed_w1[r], routed_w1_scale[r],
            routed_w3[r], routed_w3_scale[r],
            routed_w2[r], routed_w2_scale[r],
            shared_w1[r], shared_w1_scale[r],
            shared_w3[r], shared_w3_scale[r],
            shared_w2[r], shared_w2_scale[r],
            o_proj_wo_a_full[r], o_proj_wo_b_full[r],
            attn_stage[r], x_mixed[r],
            post_ffn[r], comb_ffn[r], ffn_out[r],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            stage_done,
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            r,
            device=r,
        )


# Kernel-only smoke fixtures.
def _layer_count(name):
    if name in CSA_LAYER_STACKED_NAMES:
        return CSA_NUM_LAYERS
    if name in HCA_LAYER_STACKED_NAMES:
        return HCA_NUM_LAYERS
    if name in FWD_LAYER_STACKED_NAMES:
        return FWD_NUM_LAYERS
    return 1


def _expand_rank_axis(value, torch):
    """Replicate one TP-group fixture across all EP ranks."""
    rank_count = value.shape[0]
    if rank_count == N_RANKS:
        return value.contiguous()
    if rank_count != TP_SIZE:
        raise ValueError(f"fixture rank axis must be TP={TP_SIZE} or EP={N_RANKS}, got {rank_count}")
    repeats = [N_RANKS // TP_SIZE, *([1] * (value.ndim - 1))]
    return value.repeat(*repeats).contiguous()


def _make_stacked_spec(name, base_specs, cache_block_nums=None):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    count = _layer_count(name)
    unit_shape = list(spec.shape[1:])
    if cache_block_nums and name in cache_block_nums:
        unit_shape[0] = cache_block_nums[name]
    flatten_layers = name in FLATTENED_LAYER_STACKED_NAMES
    if flatten_layers:
        packed_shape = [N_RANKS, count * unit_shape[0], *unit_shape[1:]]
    else:
        packed_shape = [N_RANKS, count, *unit_shape]

    def init_value():
        if cache_block_nums and name in cache_block_nums:
            return torch.zeros(packed_shape, dtype=spec.dtype)
        if name == "tid2eid":
            token_ids = torch.arange(VOCAB, dtype=torch.int32).view(VOCAB, 1)
            topk_ids = torch.arange(TOPK, dtype=torch.int32).view(1, TOPK)
            rows = []
            for layer in range(count):
                rows.append((token_ids * TOPK + topk_ids + layer * TOPK) % N_EXPERTS_GLOBAL)
            packed = torch.cat(rows, dim=0)
            return packed.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()
        layer_values = [
            _expand_rank_axis(_spec_value(spec, torch), torch)
            for _ in range(count)
        ]
        if flatten_layers:
            return torch.cat(layer_values, dim=1)
        return torch.stack(layer_values, dim=1)

    # Mutable caches are fixture outputs.
    return TensorSpec(
        name, packed_shape, spec.dtype,
        init_value=init_value, is_output=name in RESIDENT_CACHE_OUTPUT_NAMES,
    )


def _make_o_proj_tp_stacked_spec(name, base_specs):
    """Pack 43 resident output-projection layers in TP-sharded layout."""
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    if name == "wo_a":
        packed_shape = [N_RANKS, FWD_NUM_LAYERS * O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN]
    elif name == "wo_b":
        packed_shape = [N_RANKS, FWD_NUM_LAYERS * D, O_PROJ_LOCAL_COLS]
    else:
        raise ValueError(f"unsupported o-projection TP weight {name!r}")

    def init_value():
        packed = torch.empty(packed_shape, dtype=spec.dtype)
        for layer in range(FWD_NUM_LAYERS):
            full = _spec_value(spec, torch)
            source_ranks = full.shape[0]
            if source_ranks not in {TP_SIZE, N_RANKS}:
                raise ValueError(
                    f"{name} fixture rank axis must be TP={TP_SIZE} or "
                    f"EP={N_RANKS}, got {source_ranks}"
                )
            for rank in range(N_RANKS):
                source_rank = rank if source_ranks == N_RANKS else rank % TP_SIZE
                tp_rank = rank % TP_SIZE
                if name == "wo_a":
                    group_start = tp_rank * O_PROJ_LOCAL_GROUPS
                    layer_start = layer * O_PROJ_LOCAL_GROUPS
                    target = packed[rank, layer_start : layer_start + O_PROJ_LOCAL_GROUPS]
                    source = full[source_rank, group_start : group_start + O_PROJ_LOCAL_GROUPS]
                    target.copy_(source)
                else:
                    col_start = tp_rank * O_PROJ_LOCAL_COLS
                    row_start = layer * D
                    target = packed[rank, row_start : row_start + D]
                    source = full[source_rank, :, col_start : col_start + O_PROJ_LOCAL_COLS]
                    target.copy_(source)
        return packed

    return TensorSpec(name, packed_shape, spec.dtype, init_value=init_value, is_output=False)


def _make_shared_spec(name, base_specs):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]

    def init_value():
        return _expand_rank_axis(_spec_value(spec, torch), torch)

    return TensorSpec(name, [N_RANKS, *spec.shape[1:]], spec.dtype, init_value=init_value, is_output=False)


def _global_token_index_map(num_tiles, torch):
    """Map each rank-local row to its TP-group prompt token."""
    local_tokens = num_tiles * T
    local_row = torch.arange(local_tokens, dtype=torch.int64)
    return torch.stack([(rank % TP_SIZE) * local_tokens + local_row for rank in range(N_RANKS)], dim=0,).contiguous()


# Canonical host-tensor order for a single unified prefill layer.
HOST_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w", "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w", "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache", "ori_block_table", "ori_slot_mapping_full",
    "hca_cmp_kv", "csa_cmp_kv", "hca_cmp_block_table", "csa_cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "position_ids_local", "position_ids_full",
    "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
    "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full", "csa_state_slot_mapping_full",
    "csa_inner_state_slot_mapping_full",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "x_next",
)


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def build_single_layer_tensor_specs(start_pos=START_POS, token_count=TP_SIZE * T, layer_id=2):
    """Build the single-layer tensor specs used by the stacked forward fixtures."""
    import torch
    from golden import ScalarSpec, TensorSpec

    def kind_specs(build_fn):
        return {
            s.name: s
            for s in build_fn(start_pos=start_pos, token_count=token_count, tp_size=TP_SIZE)
            if isinstance(s, TensorSpec)
        }

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active_kind = _attention_kind_for_layer(layer_id)
    active = {"swa": swa, "hca": hca, "csa": csa}[active_kind]
    active_tokens = token_count // TP_SIZE

    # Unified names and source specs for the selected attention kind.
    attention_specs = [
        ("x_hc", active["x_hc_full"]),
        ("hc_attn_fn", active["hc_attn_fn"]), ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]), ("wq_b", active["wq_b"]), ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]), ("gamma_cq", active["gamma_cq"]), ("gamma_ckv", active["gamma_ckv"]),
        (
            "freqs_cos",
            TensorSpec(
                "freqs_cos", [TP_SIZE, 2, *swa["freqs_cos"].shape[1:]], swa["freqs_cos"].dtype,
                init_value=lambda: torch.stack(
                    (_spec_value(swa["freqs_cos"], torch), _spec_value(csa["freqs_cos"], torch)),
                    dim=1,
                ),
            ),
        ),
        (
            "freqs_sin",
            TensorSpec(
                "freqs_sin", [TP_SIZE, 2, *swa["freqs_sin"].shape[1:]], swa["freqs_sin"].dtype,
                init_value=lambda: torch.stack(
                    (_spec_value(swa["freqs_sin"], torch), _spec_value(csa["freqs_sin"], torch)),
                    dim=1,
                ),
            ),
        ),
        ("hca_cmp_wkv", hca["cmp_wkv"]), ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]), ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("hca_compress_state", hca["compress_state"]),
        ("hca_compress_state_block_table", hca["compress_state_block_table"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]), ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]), ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_compress_state", csa["compress_state"]),
        ("csa_compress_state_block_table", csa["compress_state_block_table"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]), ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]), ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]), ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]), ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("csa_inner_compress_state", csa["inner_compress_state"]),
        ("csa_inner_compress_state_block_table", csa["inner_compress_state_block_table"]),
        ("kv_cache", active["kv_cache"]),
        ("ori_block_table", active.get("ori_block_table", swa.get("block_table"))),
        ("ori_slot_mapping_full", active["ori_slot_mapping_full"]),
        ("hca_cmp_kv", hca["cmp_kv"]), ("csa_cmp_kv", csa["cmp_kv"]),
        ("hca_cmp_block_table", hca["cmp_block_table"]), ("csa_cmp_block_table", csa["cmp_block_table"]),
        ("idx_kv_cache", csa["idx_kv_cache"]), ("idx_kv_scale", csa["idx_kv_scale"]),
        ("idx_block_table", csa["idx_block_table"]),
        ("position_ids_local", active["position_ids_local"]), ("position_ids_full", active["position_ids_full"]),
        ("hca_cmp_slot_mapping_full", hca["cmp_slot_mapping_full"]),
        ("hca_state_slot_mapping_full", hca["state_slot_mapping_full"]),
        ("csa_cmp_slot_mapping_full", csa["cmp_slot_mapping_full"]),
        ("csa_idx_slot_mapping_full", csa["idx_slot_mapping_full"]),
        ("csa_state_slot_mapping_full", csa["state_slot_mapping_full"]),
        ("csa_inner_state_slot_mapping_full", csa["inner_state_slot_mapping_full"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]), ("wo_b", active["wo_b"]), ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [
        TensorSpec(name, list(src.shape), src.dtype, init_value=src.init_value, is_output=src.is_output)
        for name, src in attention_specs
    ]

    for spec in build_moe_tensor_specs(layer_id=layer_id):
        if not isinstance(spec, TensorSpec) or spec.name in {"x_hc", "x_next"}:
            continue
        if spec.name == "tid2eid":
            def init_tid2eid(spec=spec):
                _, vocab, topk = spec.shape
                ids = torch.arange(vocab, dtype=torch.int64).view(vocab, 1)
                ks = torch.arange(topk, dtype=torch.int64).view(1, topk)
                table = ((ids * topk + ks) % N_EXPERTS_GLOBAL).to(dtype=spec.dtype)
                return table.unsqueeze(0).expand(N_RANKS, -1, -1).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_tid2eid))
        elif spec.name == "input_ids":
            input_ids_shape = list(spec.shape)
            if len(input_ids_shape) != 2 or input_ids_shape[0] != N_RANKS:
                raise ValueError(f"MoE input_ids must be [EP, T], got shape {spec.shape}")

            def init_input_ids(spec=spec):
                tokens = spec.shape[-1]
                active = min(active_tokens, tokens)
                rows = []
                for rank in range(N_RANKS):
                    row = torch.roll(torch.arange(tokens, dtype=spec.dtype), shifts=rank)
                    if layer_id >= 3 and active < tokens:
                        row[active:] = -1
                    rows.append(row)
                return torch.stack(rows, dim=0).contiguous()

            tensor_specs.append(TensorSpec(spec.name, input_ids_shape, spec.dtype, init_value=init_input_ids))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.float32, is_output=True))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("num_tokens", torch.int32, active_tokens),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def build_tensor_specs(
    start_pos=0,
    num_tokens=TP_SIZE * T,
    ori_block_num=CSA_ORI_BLOCK_NUM,
    cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=IDX_CACHE_BLOCK_NUM,
    hca_state_block_num=HCA_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_STATE_BLOCK_NUM,
    inner_state_block_num=INNER_STATE_BLOCK_NUM,
):
    import torch
    from golden import TensorSpec

    if start_pos != 0:
        raise ValueError("the 43-layer prefill backbone requires start_pos=0")
    capacities = {
        "ori_block_num": (ori_block_num, CSA_ORI_BLOCK_NUM),
        "cmp_block_num": (cmp_block_num, CSA_CMP_BLOCK_NUM),
        "idx_block_num": (idx_block_num, IDX_CACHE_BLOCK_NUM),
        "hca_state_block_num": (hca_state_block_num, HCA_STATE_BLOCK_NUM),
        "csa_state_block_num": (csa_state_block_num, CSA_STATE_BLOCK_NUM),
        "inner_state_block_num": (inner_state_block_num, INNER_STATE_BLOCK_NUM),
    }
    undersized = [
        f"{name}={value} (minimum {minimum})"
        for name, (value, minimum) in capacities.items()
        if value < minimum
    ]
    if undersized:
        raise ValueError(
            "custom cache/state pools cannot be smaller than the canonical "
            f"physical layout: {', '.join(undersized)}"
        )
    tokens_per_wave = TP_SIZE * T
    if num_tokens < tokens_per_wave or num_tokens % tokens_per_wave != 0:
        raise ValueError(
            "layer-major DSA-CP requires full 128-row MoE waves: "
            f"num_tokens must be a positive multiple of TP_SIZE * {T} == {tokens_per_wave}, "
            f"got num_tokens={num_tokens}"
        )
    if num_tokens > PREFILL_GROUP_CAP:
        raise ValueError(f"TP-group tokens {num_tokens} exceed prefill capacity {PREFILL_GROUP_CAP}")

    num_tiles = num_tokens // tokens_per_wave
    global_token_indices = _global_token_index_map(num_tiles, torch)
    expected_global_indices = torch.arange(num_tokens, dtype=torch.int64)
    for group_base in range(0, N_RANKS, TP_SIZE):
        group_indices = global_token_indices[group_base : group_base + TP_SIZE]
        flat_group_indices = group_indices.reshape(-1)
        sorted_group_indices = torch.sort(flat_group_indices).values
        if not torch.equal(sorted_group_indices, expected_global_indices):
            raise ValueError(f"TP group at rank {group_base} is not a permutation " f"of 0..{num_tokens - 1}")
    fixture_seed = torch.initial_seed()

    base_specs = {
        spec.name: spec
        for spec in build_single_layer_tensor_specs(start_pos=start_pos, token_count=num_tokens, layer_id=0)
        if isinstance(spec, TensorSpec)
    }

    ordered_names = [
        "x_hc",
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "kv_cache", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
        "hca_cmp_kv", "csa_cmp_kv",
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
        "hca_compress_state",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
        "csa_compress_state",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
        "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
        "hca_compress_state_block_table", "csa_compress_state_block_table",
        "csa_inner_compress_state_block_table",
        "freqs_cos", "freqs_sin",
        "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
        "ori_slot_mapping_full", "position_ids_local", "position_ids_full", "input_ids",
        "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
        "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full",
        "csa_state_slot_mapping_full", "csa_inner_state_slot_mapping_full",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    ]

    cache_block_nums = {
        "kv_cache": ori_block_num,
        "hca_cmp_kv": cmp_block_num,
        "csa_cmp_kv": cmp_block_num,
        "idx_kv_cache": idx_block_num,
        "idx_kv_scale": idx_block_num,
        "hca_compress_state": hca_state_block_num,
        "csa_compress_state": csa_state_block_num,
        "csa_inner_compress_state": inner_state_block_num,
    }
    specs = []
    for name in ordered_names:
        if name == "x_hc":
            base = base_specs[name]
            x_hc_shape = list(base.shape)
            x_hc_shape[0] = N_RANKS
            x_hc_shape[1] = num_tokens

            def init_x_hc(tokens=num_tokens, dtype=base.dtype, seed=fixture_seed):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                global_x = torch.randn(
                    N_RANKS // TP_SIZE, tokens, HC_MULT, D,
                    generator=generator, dtype=torch.float32,
                )
                group_ids = torch.arange(N_RANKS, dtype=torch.int64) // TP_SIZE
                return (global_x[group_ids] * 0.05).to(dtype).contiguous()

            specs.append(TensorSpec(name, x_hc_shape, base.dtype, init_value=init_x_hc, is_output=True))
        elif name == "position_ids_local":
            local_tokens = num_tiles * T
            dtype = base_specs[name].dtype

            def init_position_ids_local(indices=global_token_indices, dtype=dtype):
                return (start_pos + indices).to(dtype).contiguous()

            specs.append(TensorSpec(name, [N_RANKS, local_tokens], dtype, init_value=init_position_ids_local))
        elif name == "input_ids":
            local_tokens = num_tiles * T
            dtype = base_specs[name].dtype

            def init_input_ids(indices=global_token_indices, dtype=dtype):
                group_ids = torch.arange(N_RANKS, dtype=torch.int64) // TP_SIZE
                token_ids = group_ids[:, None] * num_tokens + indices
                return (token_ids % VOCAB).to(dtype).contiguous()

            specs.append(TensorSpec(name, [N_RANKS, local_tokens], dtype, init_value=init_input_ids))
        elif name in {"wo_a", "wo_b"}:
            specs.append(_make_o_proj_tp_stacked_spec(name, base_specs))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs))
        else:
            specs.append(_make_stacked_spec(name, base_specs, cache_block_nums))

    # Resident rank shards for weights and persistent KV/state pools.
    for spec in specs:
        if spec.name == "x_hc" or spec.name in RESIDENT_WEIGHT_NAMES or spec.name in RESIDENT_CACHE_NAMES:
            spec.resident = "stacked"

    local_tokens = num_tiles * T
    stage_tokens = num_tokens
    o_proj_scratch_specs = [
        TensorSpec(
            "o_proj_wo_a_full", [N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT],
            torch.bfloat16,
            init_value=lambda: torch.zeros(
                N_RANKS, O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT,
                dtype=torch.bfloat16,
            ),
            is_output=False,
        ),
        TensorSpec(
            "o_proj_wo_b_full", [N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], torch.int8,
            init_value=lambda: torch.zeros(
                N_RANKS, O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS, dtype=torch.int8,
            ),
            is_output=False,
        ),
    ]
    for spec in o_proj_scratch_specs:
        spec.resident = "stacked"
        specs.append(spec)
    attn_stage = TensorSpec(
        "attn_stage", [N_RANKS, stage_tokens, HC_MULT, D], torch.float32,
        init_value=lambda: torch.zeros(N_RANKS, stage_tokens, HC_MULT, D, dtype=torch.float32),
        is_output=True,
    )
    attn_stage.resident = "stacked"
    specs.append(attn_stage)
    moe_stage_specs = [
        TensorSpec(
            "x_mixed", [N_RANKS, stage_tokens, D], torch.bfloat16,
            init_value=lambda: torch.zeros(N_RANKS, stage_tokens, D, dtype=torch.bfloat16),
            is_output=False,
        ),
        TensorSpec(
            "post_ffn", [N_RANKS, stage_tokens, HC_MULT], torch.float32,
            init_value=lambda: torch.zeros(N_RANKS, stage_tokens, HC_MULT, dtype=torch.float32),
            is_output=False,
        ),
        TensorSpec(
            "comb_ffn", [N_RANKS, stage_tokens, HC_MULT * HC_MULT], torch.float32,
            init_value=lambda: torch.zeros(
                N_RANKS, stage_tokens, HC_MULT * HC_MULT, dtype=torch.float32,
            ),
            is_output=False,
        ),
        TensorSpec(
            "ffn_out", [N_RANKS, local_tokens, D], torch.bfloat16,
            init_value=lambda: torch.zeros(N_RANKS, local_tokens, D, dtype=torch.bfloat16),
            is_output=False,
        ),
    ]
    for spec in moe_stage_specs:
        spec.resident = "stacked"
        specs.append(spec)
    return specs


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash DSA-CP 43-layer prefill-backbone driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument(
        "--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16],
        help="EP world size / rank count (parsed at import by moe).",
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, choices=[1, 2, 4],
        help="CP group size (parsed at import by the CP attention leaves).",
    )
    default_devices = os.environ.get("TASK_DEVICE", ",".join(str(i) for i in range(N_RANKS)))
    parser.add_argument(
        "-d", "--device", type=str, default=default_devices,
        help=f"comma-separated device ids; need at least {N_RANKS}",
    )
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument(
        "--num-tokens", type=int, default=TP_SIZE * T,
        help="Prompt tokens per TP/CP group; must be at most 8192.",
    )
    parser.add_argument("--ori-block-num", type=int, default=CSA_ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=IDX_CACHE_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=INNER_STATE_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)

    parser.add_argument(
        "--seed", type=int, default=20260824,
        help="Torch seed for reproducible runner inputs and weights.",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        parser.error(f"need at least {N_RANKS} devices, got {device_ids}")
    if TP_SIZE != args.tp:
        parser.error(f"import-time TP_SIZE must match --tp, got {TP_SIZE} vs {args.tp}")
    if N_RANKS != args.ep:
        parser.error(f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}")
    if args.ep % args.tp != 0:
        parser.error(f"EP must be divisible by TP/CP, got --ep {args.ep} and --tp {args.tp}")
    tokens_per_wave = TP_SIZE * T
    if args.num_tokens < tokens_per_wave or args.num_tokens % tokens_per_wave != 0:
        parser.error(
            "layer-major DSA-CP requires full 128-row MoE waves: "
            f"--num-tokens must be a positive multiple of {tokens_per_wave}"
        )

    import torch

    torch.manual_seed(args.seed)
    specs = build_tensor_specs(
        start_pos=args.start_pos, num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num, cmp_block_num=args.cmp_block_num, idx_block_num=args.idx_block_num,
        hca_state_block_num=args.hca_state_block_num, csa_state_block_num=args.csa_state_block_num,
        inner_state_block_num=args.inner_state_block_num,
    )

    result = run_jit(
        fn=l3_prefill_fwd,
        specs=specs,
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            ring_heap=PREFILL_RING_HEAP,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

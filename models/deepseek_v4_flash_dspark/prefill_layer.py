# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: EP2/TP2 representative single-layer golden
# ci: no-sim    # CI marker: distributed communication oracle requires real devices
"""DeepSeek-V4 Flash DSpark DSA-CP single-layer numerical oracle."""

import argparse
import os

import pypto.language as pl
import pypto.language.distributed as pld
from golden import ScalarSpec, TensorSpec, ratio_allclose, ratio_reldiff, run_jit
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
    golden_moe,
)
from hc_pre import hc_pre
from prefill_csa import (
    BLOCK_SIZE,
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
    INNER_COMPRESS_STATE_DIM,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_NUM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_MAX_BLOCKS,
    MAIN_COMPRESS_STATE_DIM as CSA_COMPRESS_STATE_DIM,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAX_SEQ_LEN,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    golden_prefill_attention_csa,
    prefill_attention_csa_cp,
)
from prefill_fwd import (
    FWD_GROUP_TOKENS_DYN,
    FWD_TOKENS_DYN,
    HOST_TENSOR_ORDER,
    O_PROJ_LOCAL_COLS,
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_COLS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    build_single_layer_tensor_specs,
    gather_o_proj_full_weights,
    prefill_moe_post,
    prefill_moe_wave,
)
from prefill_hca import (
    HCA_CMP_BLOCK_NUM,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    golden_prefill_attention_hca,
    prefill_attention_hca_cp,
)
from prefill_cp_token_allgather import PREFILL_GROUP_CAP, TP_SIZE
from prefill_swa import golden_prefill_attention_swa, prefill_attention_swa_cp


# model config
GROUP_TOKENS = TP_SIZE * T

# fixture
SUPPORTED_LAYERS = (0, 2, 3)

assert GROUP_TOKENS <= PREFILL_GROUP_CAP


@pl.jit.inline(auto_scope=False)
def _retire_o_proj_fixture_signals(
    weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    completion_tid: pl.Scalar[pl.TASK_ID],
):
    """Clear retained o-projection credits after one fixture invocation."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="o_proj_fixture_signal_retire",
        deps=[completion_tid],
    ) as retire_tid:
        self_rank = group_base + tp_rank
        reset_value = pl.cast(-1, pl.INT32)
        for source_tp in pl.range(TP_SIZE):
            if source_tp != tp_rank:
                pld.system.notify(
                    target=weight_ready, peer=self_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
                pld.system.notify(
                    target=weight_consumed, peer=self_rank,
                    offsets=[source_tp, 0], value=reset_value, op=pld.NotifyOp.AtomicAdd,
                )
    return retire_tid


@pl.jit(auto_scope=False)
def prefill_layer_attention(
    x_hc: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[128, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[[CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[4, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, INNER_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_stage: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one selected DSA-CP attention kind."""
    x_hc.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(0, FWD_TOKENS_DYN)
    position_ids_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    attn_stage.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    swa_profile = freqs_cos[0:1, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    swa_cos = pl.reshape(swa_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    swa_profile = freqs_sin[0:1, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    swa_sin = pl.reshape(swa_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    compressed_profile = freqs_cos[1:2, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    compressed_cos = pl.reshape(compressed_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    compressed_profile = freqs_sin[1:2, 0:MAX_SEQ_LEN, 0:ROPE_HEAD_DIM]
    compressed_sin = pl.reshape(compressed_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    wo_a_full = pl.create_tensor([O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], dtype=pl.BF16)
    wo_b_full = pl.create_tensor([O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], dtype=pl.INT8)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_layer_o_proj_order_init"):
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a, wo_b,
        wo_a_full, wo_b_full,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, pl.const(1, pl.INT32),
    )

    with pl.scope():
        if layer_id == 0:
            kv_cache, attn_stage, gather_signal = prefill_attention_swa_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                swa_cos, swa_sin,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                position_ids_local, position_ids_full,
                attn_sink,
                wo_a_full, wo_b_full, wo_b_scale,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank,
                o_proj_weight_dep,
            )
        elif layer_id == 2:
            attn_stage, gather_signal = prefill_attention_csa_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                compressed_cos, compressed_sin,
                csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape,
                csa_cmp_norm_w,
                csa_compress_state, csa_compress_state_block_table,
                csa_hadamard_idx, csa_idx_wq_b, csa_idx_wq_b_scale,
                csa_weights_proj,
                csa_inner_wkv, csa_inner_wgate, csa_inner_ape,
                csa_inner_norm_w,
                csa_inner_compress_state, csa_inner_compress_state_block_table,
                kv_cache, ori_block_table, ori_slot_mapping_full,
                csa_cmp_kv, csa_cmp_block_table,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                position_ids_local, position_ids_full,
                csa_cmp_slot_mapping_full, csa_idx_slot_mapping_full, csa_state_slot_mapping_full,
                csa_inner_state_slot_mapping_full,
                attn_sink,
                wo_a_full, wo_b_full, wo_b_scale,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank,
                o_proj_weight_dep,
            )
        else:
            attn_stage, gather_signal = prefill_attention_hca_cp(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w,
                wq_a, wq_b, wq_b_scale,
                wkv, gamma_cq, gamma_ckv,
                compressed_cos, compressed_sin,
                hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape,
                hca_cmp_norm_w,
                hca_compress_state, hca_compress_state_block_table,
                kv_cache, ori_slot_mapping_full, ori_block_table,
                hca_cmp_kv, hca_cmp_block_table,
                position_ids_local, position_ids_full,
                hca_cmp_slot_mapping_full, hca_state_slot_mapping_full,
                attn_sink,
                wo_a_full, wo_b_full, wo_b_scale,
                attn_stage,
                gather_window, gather_signal,
                group_base, tp_rank,
                o_proj_weight_dep,
            )

    _retire_o_proj_fixture_signals(
        o_proj_weight_ready, o_proj_weight_consumed,
        group_base, tp_rank, o_proj_weight_dep,
    )
    return attn_stage


@pl.jit.inline
def _clear_moe_fixture_signals(
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    completion_tid: pl.Scalar[pl.TASK_ID],
):
    """Clear retained MoE windows after the single-layer fixture retires."""
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="moe_fixture_signal_clear",
        deps=[completion_tid],
    ) as clear_tid:
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(arrived, [src, 0], zero)
            pl.write(data_arrived, [src, 0], zero)
            pl.write(combine_arrived, [src, 0], zero)
            pl.write(stage_done, [src, 0], zero)
    return clear_tid


@pl.jit(auto_scope=False)
def prefill_layer_moe(
    attn_stage: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
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
    x_mixed: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    post_ffn: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32]],
    comb_ffn: pl.Out[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32]],
    ffn_out: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run one canonical rank-local MoE wave from the full attention state."""
    attn_stage.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    input_ids.bind_dynamic(0, FWD_TOKENS_DYN)
    ffn_out.bind_dynamic(0, FWD_TOKENS_DYN)

    hc_pre(attn_stage, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, x_mixed, post_ffn, comb_ffn)
    tp_rank = my_rank % TP_SIZE
    x_mixed_local = x_mixed[tp_rank * T : tp_rank * T + T, 0:D]
    stage_token = pl.create_tensor([1], dtype=pl.INT32)
    barrier_tid = prefill_moe_wave(
        x_mixed_local,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        ffn_out,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        stage_done, stage_token,
        layer_id, my_rank, pl.const(1, pl.INT32),
    )
    _clear_moe_fixture_signals(arrived, data_arrived, combine_arrived, stage_done, barrier_tid)
    return ffn_out


@pl.jit(auto_scope=False)
def prefill_moe_post_fixture(
    ffn_out_local: pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16],
    residual_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    post_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32],
    comb_full: pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32],
    x_hc_full: pl.InOut[pl.Tensor[[FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pl.InOut[pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16]],
    gather_signal: pl.InOut[pld.DistributedTensor[[TP_SIZE, 1], pl.INT32]],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Gather rank-local MoE output and apply full-state HC post-mixing."""
    ffn_out_local.bind_dynamic(0, FWD_TOKENS_DYN)
    residual_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    post_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    comb_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    x_hc_full.bind_dynamic(0, FWD_GROUP_TOKENS_DYN)
    return prefill_moe_post(
        ffn_out_local, residual_full, post_full, comb_full,
        x_hc_full, gather_window, gather_signal, group_base, tp_rank,
    )


@pl.jit.host
def l3_prefill_layer(
    x_hc: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, 128, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                HCA_STATE_BLOCK_NUM,
                HCA_STATE_BLOCK_SIZE,
                HCA_COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, 4, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                CSA_STATE_BLOCK_NUM,
                CSA_STATE_BLOCK_SIZE,
                CSA_COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, 4, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                INNER_STATE_BLOCK_NUM,
                INNER_STATE_BLOCK_SIZE,
                INNER_COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8,]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, IDX_CACHE_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT32],
    hca_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping_full: pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN], pl.INT64],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    attn_stage: pl.InOut[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    x_mixed: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, D], pl.BF16]],
    post_ffn: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT], pl.FP32]],
    comb_ffn: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT * HC_MULT], pl.FP32]],
    ffn_out: pl.Out[pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, FWD_GROUP_TOKENS_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
):
    """Run one DSA-CP layer across all ranks."""
    x_hc.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    position_ids_local.bind_dynamic(1, FWD_TOKENS_DYN)
    position_ids_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    hca_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_cmp_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_idx_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    csa_inner_state_slot_mapping_full.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    attn_stage.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    x_mixed.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    post_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    comb_ffn.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
    ffn_out.bind_dynamic(1, FWD_TOKENS_DYN)
    x_next.bind_dynamic(1, FWD_GROUP_TOKENS_DYN)
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

    for rank in pl.range(pld.world_size()):
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
        group_base = rank // TP_SIZE * TP_SIZE
        tp_rank = rank % TP_SIZE
        prefill_layer_attention(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank],
            wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank],
            hca_cmp_norm_w[rank],
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank],
            csa_cmp_norm_w[rank],
            csa_compress_state[rank], csa_compress_state_block_table[rank],
            csa_hadamard_idx[rank], csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank],
            csa_weights_proj[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank],
            csa_inner_norm_w[rank],
            csa_inner_compress_state[rank], csa_inner_compress_state_block_table[rank],
            kv_cache[rank], ori_block_table[rank], ori_slot_mapping_full[rank],
            hca_cmp_kv[rank], csa_cmp_kv[rank],
            hca_cmp_block_table[rank], csa_cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            position_ids_local[rank], position_ids_full[rank],
            hca_cmp_slot_mapping_full[rank], hca_state_slot_mapping_full[rank],
            csa_cmp_slot_mapping_full[rank], csa_idx_slot_mapping_full[rank], csa_state_slot_mapping_full[rank],
            csa_inner_state_slot_mapping_full[rank],
            attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            attn_stage[rank],
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            group_base, tp_rank,
            layer_id,
            device=rank,
        )

    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        stage_done = pld.window(stage_done_buf, [N_RANKS, 1], dtype=pl.INT32)
        prefill_layer_moe(
            attn_stage[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank],
            gate_w[rank], gate_bias[rank],
            tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_mixed[rank], post_ffn[rank], comb_ffn[rank], ffn_out[rank],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            stage_done,
            layer_id, rank,
            device=rank,
        )

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        group_base = rank // TP_SIZE * TP_SIZE
        tp_rank = rank % TP_SIZE
        prefill_moe_post_fixture(
            ffn_out[rank],
            attn_stage[rank], post_ffn[rank], comb_ffn[rank],
            x_next[rank],
            gather_window, gather_signal,
            group_base, tp_rank,
            device=rank,
        )


_RESIDENT_WEIGHT_NAMES = frozenset(
    {
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
        "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale",
        "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
        "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape",
        "hca_cmp_norm_w",
        "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape",
        "csa_cmp_norm_w",
        "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale",
        "csa_weights_proj",
        "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape",
        "csa_inner_norm_w",
        "attn_sink",
        "wo_a", "wo_b", "wo_b_scale",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
        "norm_w",
        "gate_w", "gate_bias",
        "tid2eid",
        "routed_w1", "routed_w1_scale",
        "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale",
        "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
    }
)

_CACHE_OUTPUTS = {
    0: {"kv_cache"},
    2: {
        "kv_cache", "csa_cmp_kv", "csa_compress_state",
        "csa_inner_compress_state", "idx_kv_cache", "idx_kv_scale",
    },
    3: {"kv_cache", "hca_cmp_kv", "hca_compress_state"},
}

_CACHE_STATE_NAMES = frozenset().union(*_CACHE_OUTPUTS.values())

_STAGE_TENSOR_ORDER = ("attn_stage", "x_mixed", "post_ffn", "comb_ffn", "ffn_out")


def _expand_rank_axis(value, torch):
    if value.shape[0] == N_RANKS:
        return value.contiguous()
    if value.shape[0] != TP_SIZE:
        raise ValueError(f"fixture rank axis must be TP={TP_SIZE} or EP={N_RANKS}, got {value.shape[0]}")
    repeats = [N_RANKS // TP_SIZE, *([1] * (value.ndim - 1))]
    return value.repeat(*repeats).contiguous()


def _materialize_spec(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def build_tensor_specs(layer_id=0):
    """Build one full DSA-CP layer boundary from the canonical leaf fixtures."""
    import torch

    if layer_id not in SUPPORTED_LAYERS:
        raise ValueError(f"layer_id must be one of {SUPPORTED_LAYERS}, got {layer_id}")
    single_layer_specs = build_single_layer_tensor_specs(start_pos=0, token_count=GROUP_TOKENS, layer_id=layer_id)
    base_specs = {spec.name: spec for spec in single_layer_specs if isinstance(spec, TensorSpec)}

    def init_full_x():
        full = _materialize_spec(base_specs["x_hc"], torch)[0]
        groups = []
        for group_id in range(N_RANKS // TP_SIZE):
            group_full = torch.roll(full, shifts=group_id, dims=0)
            group_ranked = group_full.unsqueeze(0).expand(TP_SIZE, -1, -1, -1).contiguous()
            groups.append(group_ranked)
        return torch.cat(groups, dim=0)

    specs_by_name = {
        "x_hc": TensorSpec("x_hc", [N_RANKS, GROUP_TOKENS, HC_MULT, D], torch.float32, init_value=init_full_x)
    }
    for name in HOST_TENSOR_ORDER:
        if name in {"x_hc", "x_next"}:
            continue
        source = base_specs[name]

        if name in {"wo_a", "wo_b"}:
            split_dim = 0 if name == "wo_a" else 1

            def init_o_proj_shards(source=source, split_dim=split_dim):
                full_weight = _materialize_spec(source, torch)[0]
                tp_shards = torch.stack(torch.chunk(full_weight, TP_SIZE, dim=split_dim), dim=0)
                repeats = [N_RANKS // TP_SIZE, *([1] * (tp_shards.ndim - 1))]
                return tp_shards.repeat(*repeats).contiguous()

            local_shape = list(source.shape[1:])
            local_shape[split_dim] //= TP_SIZE
            specs_by_name[name] = TensorSpec(name, [N_RANKS, *local_shape], source.dtype, init_value=init_o_proj_shards)
            continue

        def init_ranked(source=source):
            return _expand_rank_axis(_materialize_spec(source, torch), torch)

        shape = list(source.shape)
        shape[0] = N_RANKS
        specs_by_name[name] = TensorSpec(
            name, shape, source.dtype,
            init_value=init_ranked, is_output=name in _CACHE_STATE_NAMES,
        )

    def init_attn_stage():
        return torch.zeros(N_RANKS, GROUP_TOKENS, HC_MULT, D, dtype=torch.float32)

    stage_specs = [
        TensorSpec(
            "attn_stage", [N_RANKS, GROUP_TOKENS, HC_MULT, D], torch.float32,
            init_value=init_attn_stage, is_output=True,
        ),
        TensorSpec("x_mixed", [N_RANKS, GROUP_TOKENS, D], torch.bfloat16, is_output=True),
        TensorSpec("post_ffn", [N_RANKS, GROUP_TOKENS, HC_MULT], torch.float32, is_output=True),
        TensorSpec(
            "comb_ffn", [N_RANKS, GROUP_TOKENS, HC_MULT * HC_MULT], torch.float32,
            is_output=True,
        ),
        TensorSpec("ffn_out", [N_RANKS, T, D], torch.bfloat16, is_output=True),
    ]
    specs_by_name.update({spec.name: spec for spec in stage_specs})
    specs_by_name["x_next"] = TensorSpec("x_next", [N_RANKS, GROUP_TOKENS, HC_MULT, D], torch.float32, is_output=True,)

    for name in _RESIDENT_WEIGHT_NAMES:
        specs_by_name[name].resident = "stacked"

    tensor_order = (*[name for name in HOST_TENSOR_ORDER if name != "x_next"], *_STAGE_TENSOR_ORDER, "x_next")
    return [specs_by_name[name] for name in tensor_order] + [ScalarSpec("layer_id", torch.int32, layer_id)]


def _attention_golden_tensors(tensors, rank, layer_id, x_out):
    wo_a_full = tensors["wo_a"][rank : rank + TP_SIZE].reshape(O_GROUPS, O_LORA, O_GROUP_IN)
    wo_b_full = tensors["wo_b"][rank : rank + TP_SIZE].permute(1, 0, 2).reshape(D, O_GROUPS * O_LORA)
    common = {
        "x_hc": tensors["x_hc"][rank],
        "hc_attn_fn": tensors["hc_attn_fn"][rank],
        "hc_attn_scale": tensors["hc_attn_scale"][rank],
        "hc_attn_base": tensors["hc_attn_base"][rank],
        "attn_norm_w": tensors["attn_norm_w"][rank],
        "wq_a": tensors["wq_a"][rank],
        "wq_b": tensors["wq_b"][rank],
        "wq_b_scale": tensors["wq_b_scale"][rank],
        "wkv": tensors["wkv"][rank],
        "gamma_cq": tensors["gamma_cq"][rank],
        "gamma_ckv": tensors["gamma_ckv"][rank],
        "freqs_cos": tensors["freqs_cos"][rank, 1 if layer_id in (2, 3) else 0],
        "freqs_sin": tensors["freqs_sin"][rank, 1 if layer_id in (2, 3) else 0],
        "kv_cache": tensors["kv_cache"][rank],
        "ori_block_table": tensors["ori_block_table"][rank],
        "ori_slot_mapping": tensors["ori_slot_mapping_full"][rank],
        "position_ids": tensors["position_ids_full"][rank],
        "attn_sink": tensors["attn_sink"][rank],
        "wo_a": wo_a_full,
        "wo_b": wo_b_full,
        "wo_b_scale": tensors["wo_b_scale"][rank],
        "x_out": x_out,
        "num_tokens": GROUP_TOKENS,
    }
    if layer_id == 0:
        common["block_table"] = common.pop("ori_block_table")
        return common
    if layer_id == 3:
        common.update(
            {
                "cmp_wkv": tensors["hca_cmp_wkv"][rank],
                "cmp_wgate": tensors["hca_cmp_wgate"][rank],
                "cmp_ape": tensors["hca_cmp_ape"][rank],
                "cmp_norm_w": tensors["hca_cmp_norm_w"][rank],
                "compress_state": tensors["hca_compress_state"][rank],
                "compress_state_block_table": tensors["hca_compress_state_block_table"][rank],
                "cmp_kv": tensors["hca_cmp_kv"][rank],
                "cmp_block_table": tensors["hca_cmp_block_table"][rank],
                "cmp_slot_mapping": tensors["hca_cmp_slot_mapping_full"][rank],
                "state_slot_mapping": tensors["hca_state_slot_mapping_full"][rank],
            }
        )
        return common
    common.update(
        {
            "cmp_wkv": tensors["csa_cmp_wkv"][rank],
            "cmp_wgate": tensors["csa_cmp_wgate"][rank],
            "cmp_ape": tensors["csa_cmp_ape"][rank],
            "cmp_norm_w": tensors["csa_cmp_norm_w"][rank],
            "compress_state": tensors["csa_compress_state"][rank],
            "compress_state_block_table": tensors["csa_compress_state_block_table"][rank],
            "hadamard_idx": tensors["csa_hadamard_idx"][rank],
            "idx_wq_b": tensors["csa_idx_wq_b"][rank],
            "idx_wq_b_scale": tensors["csa_idx_wq_b_scale"][rank],
            "idx_weights_proj": tensors["csa_weights_proj"][rank],
            "inner_wkv": tensors["csa_inner_wkv"][rank],
            "inner_wgate": tensors["csa_inner_wgate"][rank],
            "inner_ape": tensors["csa_inner_ape"][rank],
            "inner_norm_w": tensors["csa_inner_norm_w"][rank],
            "inner_compress_state": tensors["csa_inner_compress_state"][rank],
            "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"][rank],
            "cmp_kv": tensors["csa_cmp_kv"][rank],
            "cmp_block_table": tensors["csa_cmp_block_table"][rank],
            "idx_kv_cache": tensors["idx_kv_cache"][rank],
            "idx_kv_scale": tensors["idx_kv_scale"][rank],
            "idx_block_table": tensors["idx_block_table"][rank],
            "cmp_slot_mapping": tensors["csa_cmp_slot_mapping_full"][rank],
            "idx_slot_mapping": tensors["csa_idx_slot_mapping_full"][rank],
            "state_slot_mapping": tensors["csa_state_slot_mapping_full"][rank],
            "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping_full"][rank],
        }
    )
    return common


def golden_prefill_layer(tensors):
    """Compute the full-attention and distributed-MoE reference."""
    import torch

    from hc_pre import golden_hc_pre

    layer_id = int(tensors["layer_id"])
    attention_golden = {
        0: golden_prefill_attention_swa,
        2: golden_prefill_attention_csa,
        3: golden_prefill_attention_hca,
    }[layer_id]
    cache_outputs = _CACHE_OUTPUTS[layer_id]

    for group_base in range(0, N_RANKS, TP_SIZE):
        full_attn = torch.zeros(GROUP_TOKENS, HC_MULT, D, dtype=torch.float32)
        attention_tensors = _attention_golden_tensors(tensors, group_base, layer_id, full_attn)
        attention_golden(attention_tensors)
        full_attn = attention_tensors["x_out"]
        for rank in range(group_base, group_base + TP_SIZE):
            tensors["attn_stage"][rank].copy_(full_attn)
            for name in cache_outputs:
                tensors[name][rank].copy_(tensors[name][group_base])

    local_attn = torch.empty(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        golden_hc_pre({
            "x": tensors["attn_stage"][rank],
            "hc_fn": tensors["hc_ffn_fn"][rank],
            "hc_scale": tensors["hc_ffn_scale"][rank],
            "hc_base": tensors["hc_ffn_base"][rank],
            "x_mixed": tensors["x_mixed"][rank],
            "post": tensors["post_ffn"][rank],
            "comb": tensors["comb_ffn"][rank],
        })
        tp_rank = rank % TP_SIZE
        local_attn[rank].copy_(tensors["attn_stage"][rank, tp_rank * T : (tp_rank + 1) * T])

    local_next = torch.zeros_like(local_attn)
    moe_tensors = dict(tensors)
    moe_tensors.update({"x_hc": local_attn, "x_next": local_next, "num_tokens": T, "layer_id": layer_id})
    golden_moe(moe_tensors)

    for group_base in range(0, N_RANKS, TP_SIZE):
        full_next = local_next[group_base : group_base + TP_SIZE].reshape(GROUP_TOKENS, HC_MULT, D)
        for rank in range(group_base, group_base + TP_SIZE):
            tensors["x_next"][rank].copy_(full_next)


def _compare_functions(layer_id):
    attention_diff_thd = 3e-3 if layer_id == 0 else 5e-3
    compare = {
        "attn_stage": ratio_reldiff(diff_thd=attention_diff_thd, pct_thd=0.005, max_diff_hd=1),
        "x_mixed": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "post_ffn": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "comb_ffn": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "ffn_out": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
    }
    if layer_id == 3:
        compare.update(
            {
                "hca_cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005),
                "hca_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.005),
            }
        )
    elif layer_id == 2:
        compare.update(
            {
                "csa_cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005),
                "csa_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.005),
                "csa_inner_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.005),
                "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
                "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
            }
        )
    return compare


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash DSpark DSA-CP single-layer golden.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8, 16])
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=[1, 2, 4])
    parser.add_argument("--layer-id", type=int, default=0, choices=list(SUPPORTED_LAYERS))
    parser.add_argument(
        "-d", "--device",
        default=os.environ.get("TASK_DEVICE", ",".join(str(index) for index in range(N_RANKS))),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir", default=None)
    args = parser.parse_args()

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < N_RANKS:
        parser.error(f"need at least {N_RANKS} devices, got {device_ids}")
    if TP_SIZE != args.tp:
        parser.error(f"import-time TP_SIZE must match --tp, got {TP_SIZE} vs {args.tp}")
    if N_RANKS != args.ep:
        parser.error(f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}")
    if args.ep % args.tp != 0:
        parser.error(f"EP must be divisible by TP/CP, got --ep {args.ep} and --tp {args.tp}")

    import torch

    torch.manual_seed(args.seed)
    result = run_jit(
        fn=l3_prefill_layer,
        specs=build_tensor_specs(args.layer_id),
        golden_fn=golden_prefill_layer,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS], num_sub_workers=0
            ),
        ),
        runtime_cfg=dict(platform=args.platform),
        rtol=1e-3,
        atol=1e-3,
        compare_fn=_compare_functions(args.layer_id),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

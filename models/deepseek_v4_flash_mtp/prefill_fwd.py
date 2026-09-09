# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""DeepSeek-V4 Flash prefill entry with lib-owned context parallelism.

The serving-facing ``l3_prefill_fwd`` ABI owns request distribution and CP
workspace. Cold single-owner requests use CP; existing prefix/chunk and
multiple-owner requests retain local scheduling. CP=1 uses local scheduling
while preserving the configured expert and LM-head parallel groups.

Both schedules follow model order through shared attention math and MoE,
then HC head and final RMSNorm. The HOST projects selected hidden rows through
the established LM head. The colocated standalone runner is an execution
smoke check, not a numerical oracle.
"""

import argparse

import pypto.language as pl
import pypto.language.distributed as pld
from golden import run
from pypto.ir import DistributedConfig

# prefill_fwd is self-contained: it imports kernels, constants, and per-kind
# spec builders directly from the leaf modules (no dependency on prefill_layer).
# The MoE slab matches the prefill attention tile.
import config
# Import moe first: it applies the EP/FLASH override before the attention modules
# bake config-derived MoE shapes (matches prefill_layer's import order).
from moe import (
    PrefillMoELayout,
    make_prefill_moe,
    D,
    HC_DIM,
    HC_MULT,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_prefill_moe_signals,
    check_prefill_moe_slab,
    PREFILL_MOE_EXPERT_SCALE_PAD,
    PREFILL_MOE_SCALE_PAD,
)

T = config.PREFILL_TOKENS
PREFILL_MOE_LAYOUT = PrefillMoELayout(T)
prefill_moe = make_prefill_moe(PREFILL_MOE_LAYOUT)
PREFILL_MOE_ROUTES_PER_SRC = PREFILL_MOE_LAYOUT.routes_per_source
PREFILL_MOE_TOTAL_CAP = PREFILL_MOE_LAYOUT.total_capacity
PREFILL_MOE_GROUPED_TOTAL_CAP = PREFILL_MOE_LAYOUT.grouped_capacity
from config import FLASH as MODEL_CONFIG
from prefill_swa import (
    build_tensor_specs as build_swa_attention_tensor_specs,
    prefill_attention_swa,
)
from prefill_hca import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAIN_OUT_DIM as HCA_MAIN_OUT_DIM,
    build_tensor_specs as build_hca_attention_tensor_specs,
    prefill_attention_hca,
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
    PREFILL_IDX_BLOCK_NUM,
    Q_LORA,
    ROPE_HEAD_DIM,
    SPARSE_CMP_MAX_BLOCKS,
    SPARSE_ORI_MAX_BLOCKS,
    START_POS,
    build_tensor_specs as build_csa_attention_tensor_specs,
    prefill_attention_csa,
)
from hc_head import hc_head
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    lm_head_test,
)
from rmsnorm import rms_norm

assert config.PREFILL_TOKENS == T
check_prefill_moe_slab(T)


# ---------------------------------------------------------------------------
# Model layer schedule (DeepSeek-V4 Flash, 43 hidden layers):
#   layer 0, 1                     -> swa
#   layer 2, 4, ..., 40            -> csa   (20 layers, loop body)
#   layer 3, 5, ..., 41            -> hca   (20 layers, loop body)
#   layer 42 (FWD_LAST_LAYER)      -> csa   (final layer)
# CSA total = 20 (loop) + 1 (last) = 21 ; HCA total = 20.
# ---------------------------------------------------------------------------
MODEL_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_NUM_LAYERS = 43
CSA_NUM_LAYERS = 21
HCA_NUM_LAYERS = 20
HCA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // HCA_COMPRESS_RATIO
CSA_CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // CSA_COMPRESS_RATIO
HCA_COMPRESS_STATE_DIM = 2 * HCA_MAIN_OUT_DIM
CSA_COMPRESS_STATE_DIM = 2 * CSA_MAIN_OUT_DIM
CSA_INNER_COMPRESS_STATE_DIM = 2 * INNER_OUT_DIM
FWD_LAST_LAYER = FWD_NUM_LAYERS - 1
CSA_LAST_ORDER = CSA_NUM_LAYERS - 1

FWD_TOKENS_DYN = pl.dynamic("PREFILL_FWD_TOKENS_DYN")
LAST_MOE_EPOCH = 2 * HCA_NUM_LAYERS + 3
PRE_HC_COPY_TOKEN_TILE = 4
assert T % PRE_HC_COPY_TOKEN_TILE == 0

# The LM head owns its barrier counters, so its epoch restarts at 1 rather
# than continuing the MoE numbering.
LM_HEAD_COMM_EPOCH = 1
assert MODEL_NUM_LAYERS == 43, "DeepSeek-V4 Flash hidden layer count changed"

# Physical cache pools are runtime-sized.  The first dimension of each
# stacked cache is the per-layer pool size multiplied by its layer count.
FWD_ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
FWD_HCA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
FWD_CSA_CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_CMP_BLOCK_NUM_DYN")
FWD_IDX_BLOCK_NUM_DYN = pl.dynamic("PREFILL_IDX_BLOCK_NUM_DYN")
FWD_HCA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")
FWD_CSA_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CSA_STATE_BLOCK_NUM_DYN")
FWD_INNER_STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_INNER_STATE_BLOCK_NUM_DYN")

# Replicated head weights (per-rank, not layer-stacked): hc_head projection and
# the final RMSNorm gamma — mirrors decode_fwd.
HC_HEAD_NAMES = ["hc_head_fn", "hc_head_scale", "hc_head_base"]
FINAL_NORM_NAMES = ["final_norm_w"]

# Per-FWD-layer stacked weights (sliced by the FWD layer index 0..42).
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
# CSA-compact stacked weights (sliced by the CSA order index 0..20).
CSA_LAYER_STACKED_NAMES = [
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx", "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_inner_compress_state", "csa_cmp_kv", "idx_kv_cache", "idx_kv_scale",
]
# HCA-compact stacked weights (sliced by the HCA order index 0..19).
HCA_LAYER_STACKED_NAMES = [
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_cmp_kv",
]
# Replicated once and passed whole to every layer (block tables are smoke zeros;
# slot mappings depend only on token position + a fixed per-kind compress ratio,
# so a single copy per name is shared across all layers of that kind).
SHARED_NAMES = [
    "freqs_cos", "freqs_sin",
    "ori_block_table", "hca_cmp_block_table", "csa_cmp_block_table", "idx_block_table",
    "hca_compress_state_block_table", "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "ori_slot_mapping", "position_ids", "input_ids",
    "hca_cmp_slot_mapping", "hca_state_slot_mapping",
    "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
    "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
]

# KV / state caches: per-token persistent buffers, not weights — kept as host
# tensors (re-bound each dispatch) rather than device-resident.
CACHE_NAMES = {
    "kv_cache", "hca_cmp_kv", "csa_cmp_kv",
    "hca_compress_state", "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
}

# Static weight parameters to keep device-resident, sharded per rank. Every host
# param is a leading-dim-stacked ``[N_RANKS, *tail]`` tensor the orchestrator
# slices as ``weight[r]`` and dispatches to ``device=r``; marking these
# resident="stacked" makes the harness upload shard ``r`` to card ``r`` once (via
# ``alloc_stacked_tensor``) and reuse it across dispatches, skipping the
# per-dispatch H2D/D2H. Covers every stacked attention / MoE weight, the per-kind
# compressor weights, the replicated head weights, and the constant RoPE tables —
# but NOT the KV/state caches (``CACHE_NAMES``) nor the per-step metadata (slot
# mappings, block tables, ids, sparse indices), which change per token.
RESIDENT_WEIGHT_NAMES = frozenset(
    [
        n
        for n in (*FWD_LAYER_STACKED_NAMES, *CSA_LAYER_STACKED_NAMES, *HCA_LAYER_STACKED_NAMES)
        if n not in CACHE_NAMES
    ]
    + ["freqs_cos", "freqs_sin"]
    + HC_HEAD_NAMES
    + FINAL_NORM_NAMES
)

# KV / state caches to keep device-resident (child_memory) as well, skipping the
# per-dispatch H2D these otherwise pay every dispatch (they dominate the residual
# host-transfer cost). All of CACHE_NAMES becomes resident.
RESIDENT_CACHE_NAMES = frozenset(CACHE_NAMES)

# Every cache in this set is mutated by one of the packed attention/compressor
# kernels and must remain visible to the following decode invocation.
RESIDENT_CACHE_OUTPUT_NAMES = RESIDENT_CACHE_NAMES


@pl.jit.inline(auto_scope=False)
def prefill_fwd(
    x_hc: pl.Tensor[[FWD_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
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
    ori_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    position_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_out: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]:
    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0]
    )
    compressed_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0]
    )
    compressed_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(
        freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [1, 0, 0]
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
    total_nt: pl.Scalar[pl.INT32] = pl.cast(0, pl.INT32)
    for owner_rank in pl.range(N_RANKS):
        total_nt = pl.max(total_nt, pl.read(num_tokens_per_owner, [owner_rank]))
    owner_nt = pl.read(num_tokens_per_owner, [my_rank])
    owner_tail_start = pl.max(owner_nt - T, pl.cast(0, pl.INT32))
    ori_block_num = pl.tensor.dim(kv_cache, 0) // FWD_NUM_LAYERS
    hca_cmp_block_num = pl.tensor.dim(hca_cmp_kv, 0) // HCA_NUM_LAYERS
    csa_cmp_block_num = pl.tensor.dim(csa_cmp_kv, 0) // CSA_NUM_LAYERS
    hca_state_block_num = pl.tensor.dim(hca_compress_state, 0) // HCA_NUM_LAYERS
    csa_state_block_num = pl.tensor.dim(csa_compress_state, 0) // CSA_NUM_LAYERS
    inner_state_block_num = pl.tensor.dim(csa_inner_compress_state, 0) // CSA_NUM_LAYERS
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0) // CSA_NUM_LAYERS
    # Keep the hand-written layer schedule readable while making every cache
    # slice use the runtime per-layer pool size.
    CSA_ORI_BLOCK_NUM = ori_block_num
    HCA_STATE_BLOCK_NUM = hca_state_block_num
    CSA_STATE_BLOCK_NUM = csa_state_block_num
    INNER_STATE_BLOCK_NUM = inner_state_block_num
    PREFILL_IDX_BLOCK_NUM = idx_block_num
    t_dim = pl.tensor.dim(x_hc, 0)
    pre_hc_hidden_tile = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    # Paged caches are slot-mapped, so tiles accumulate in place on device.
    for tile_base in pl.range(0, t_dim, T):
        with pl.scope():
            moe_x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
            moe_post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
            moe_comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
            moe_ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
            moe_dense_x = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
            moe_dense_scale = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
            moe_grouped_x = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.INT8)
            moe_grouped_scale = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, PREFILL_MOE_EXPERT_SCALE_PAD], dtype=pl.FP32)
            moe_grouped_y = pl.create_tensor([PREFILL_MOE_GROUPED_TOTAL_CAP, D], dtype=pl.BF16)
            moe_dense_y = pl.create_tensor([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16)
            moe_returned_y = pl.create_tensor([PREFILL_MOE_ROUTES_PER_SRC, D], dtype=pl.BF16)
            # Epoch numbering continues across tiles; arrival counters are monotonic.
            epoch_base: pl.Scalar[pl.INT32] = pl.cast(tile_base // T * LAST_MOE_EPOCH, pl.INT32)
            nt = total_nt - tile_base
            nt = pl.max(pl.cast(1, pl.INT32), nt)
            nt = pl.min(nt, pl.cast(T, pl.INT32))
            x_hc_tile = pl.slice(x_hc, [T, HC_MULT, D], [tile_base, 0, 0])
            ori_slot_mapping_tile = pl.slice(ori_slot_mapping, [T], [tile_base])
            position_ids_tile = pl.slice(position_ids, [T], [tile_base])
            input_ids_tile = pl.slice(input_ids, [T], [tile_base])
            hca_cmp_slot_mapping_tile = pl.slice(hca_cmp_slot_mapping, [T], [tile_base])
            hca_state_slot_mapping_tile = pl.slice(hca_state_slot_mapping, [T], [tile_base])
            csa_cmp_slot_mapping_tile = pl.slice(csa_cmp_slot_mapping, [T], [tile_base])
            csa_idx_slot_mapping_tile = pl.slice(csa_idx_slot_mapping, [T], [tile_base])
            csa_state_slot_mapping_tile = pl.slice(csa_state_slot_mapping, [T], [tile_base])
            csa_inner_state_slot_mapping_tile = pl.slice(csa_inner_state_slot_mapping, [T], [tile_base])
            x_out_tile = pl.slice(x_out, [T, D], [tile_base, 0])
            # One chronological layer loop; each layer owns its cache/state slice.
            # MoE completion orders the next attention read before the shared
            # pre-HC output storage is overwritten by the following layer.
            layer_hidden = x_hc_tile
            for layer_id in pl.range(FWD_NUM_LAYERS):
                layer_index: pl.Scalar[pl.INT32] = pl.cast(layer_id, pl.INT32)
                hc_attn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0])
                hc_attn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [layer_index * 3])
                hc_attn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [layer_index * MIX_HC])
                attn_norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [layer_index * D])
                wq_a_layer: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [layer_index * D, 0])
                wq_b_layer: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [layer_index * Q_LORA, 0])
                wq_b_scale_layer: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [layer_index * H * HEAD_DIM])
                wkv_layer: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [layer_index * D, 0])
                gamma_cq_layer: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [layer_index * Q_LORA])
                gamma_ckv_layer: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [layer_index * HEAD_DIM])
                kv_cache_layer: pl.Tensor[[CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16] = pl.slice(kv_cache, [CSA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], [layer_index * CSA_ORI_BLOCK_NUM, 0, 0, 0])
                attn_sink_layer: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [layer_index * H])
                wo_a_layer: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [layer_index * O_GROUPS, 0, 0])
                wo_b_layer: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [layer_index * D, 0])
                wo_b_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [layer_index * D])
                hc_ffn_fn_layer: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [layer_index * MIX_HC, 0])
                hc_ffn_scale_layer: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [layer_index * 3])
                hc_ffn_base_layer: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [layer_index * MIX_HC])
                norm_w_layer: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [layer_index * D])
                gate_w_layer: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [layer_index * N_EXPERTS_GLOBAL, 0])
                gate_bias_layer: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [layer_index * N_EXPERTS_GLOBAL])
                tid2eid_layer: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [layer_index * VOCAB, 0])
                routed_w1_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0])
                routed_w1_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0])
                routed_w3_layer: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [layer_index * N_LOCAL, 0, 0])
                routed_w3_scale_layer: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [layer_index * N_LOCAL, 0])
                routed_w2_layer: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [layer_index * N_LOCAL, 0, 0])
                routed_w2_scale_layer: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [layer_index * N_LOCAL, 0])
                shared_w1_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [layer_index * MOE_INTER, 0])
                shared_w1_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [layer_index * MOE_INTER])
                shared_w3_layer: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [layer_index * MOE_INTER, 0])
                shared_w3_scale_layer: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [layer_index * MOE_INTER])
                shared_w2_layer: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [layer_index * D, 0])
                shared_w2_scale_layer: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [layer_index * D])
                x_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
                with pl.scope():
                    if layer_index < 2:
                        prefill_attention_swa(
                            layer_hidden, hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                            attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer, wkv_layer,
                            gamma_cq_layer, gamma_ckv_layer, swa_freqs_cos, swa_freqs_sin, kv_cache_layer,
                            ori_block_table, ori_slot_mapping_tile, position_ids_tile, attn_sink_layer,
                            wo_a_layer, wo_b_layer, wo_b_scale_layer, x_attn, nt,
                        )
                    elif layer_index % 2 == 0:
                        type_index = (layer_index - 2) // 2
                        csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [type_index * CSA_MAIN_OUT_DIM, 0])
                        csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [type_index * CSA_MAIN_OUT_DIM, 0])
                        csa_cmp_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [type_index * CSA_COMPRESS_RATIO, 0])
                        csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM])
                        csa_compress_state_csa: pl.Tensor[[CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32] = pl.slice(csa_compress_state, [CSA_STATE_BLOCK_NUM, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], [type_index * CSA_STATE_BLOCK_NUM, 0, 0])
                        csa_hadamard_idx_csa: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM, 0])
                        csa_idx_wq_b_csa: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(csa_idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [type_index * Q_LORA, 0])
                        csa_idx_wq_b_scale_csa: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(csa_idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [type_index * IDX_N_HEADS * IDX_HEAD_DIM])
                        csa_weights_proj_csa: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(csa_weights_proj, [D, IDX_N_HEADS], [type_index * D, 0])
                        csa_inner_wkv_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [INNER_OUT_DIM, D], [type_index * INNER_OUT_DIM, 0])
                        csa_inner_wgate_csa: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [INNER_OUT_DIM, D], [type_index * INNER_OUT_DIM, 0])
                        csa_inner_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, INNER_OUT_DIM], [type_index * CSA_COMPRESS_RATIO, 0])
                        csa_inner_norm_w_csa: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [type_index * IDX_HEAD_DIM])
                        csa_inner_compress_state_csa: pl.Tensor[[INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32] = pl.slice(csa_inner_compress_state, [INNER_STATE_BLOCK_NUM, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], [type_index * INNER_STATE_BLOCK_NUM, 0, 0])
                        cmp_kv_csa = pl.slice(csa_cmp_kv, [csa_cmp_block_num, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], [type_index * csa_cmp_block_num, 0, 0, 0])
                        idx_kv_cache_csa = pl.slice(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], [type_index * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
                        idx_kv_scale_csa = pl.slice(idx_kv_scale, [PREFILL_IDX_BLOCK_NUM, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], [type_index * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
                        prefill_attention_csa(
                            layer_hidden, hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                            attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer, wkv_layer,
                            gamma_cq_layer, gamma_ckv_layer, compressed_freqs_cos, compressed_freqs_sin,
                            csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                            csa_compress_state_csa, csa_compress_state_block_table, csa_hadamard_idx_csa,
                            csa_idx_wq_b_csa, csa_idx_wq_b_scale_csa, csa_weights_proj_csa,
                            csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                            csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                            kv_cache_layer, ori_block_table, ori_slot_mapping_tile, cmp_kv_csa,
                            csa_cmp_block_table, idx_kv_cache_csa, idx_kv_scale_csa, idx_block_table,
                            position_ids_tile, csa_cmp_slot_mapping_tile, csa_idx_slot_mapping_tile,
                            csa_state_slot_mapping_tile, csa_inner_state_slot_mapping_tile, attn_sink_layer,
                            wo_a_layer, wo_b_layer, wo_b_scale_layer, x_attn, nt,
                        )
                    else:
                        type_index = (layer_index - 3) // 2
                        hca_cmp_wkv_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [type_index * HCA_MAIN_OUT_DIM, 0])
                        hca_cmp_wgate_hca: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [type_index * HCA_MAIN_OUT_DIM, 0])
                        hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32] = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [type_index * HCA_COMPRESS_RATIO, 0])
                        hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [type_index * HEAD_DIM])
                        hca_compress_state_hca: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32] = pl.slice(hca_compress_state, [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], [type_index * HCA_STATE_BLOCK_NUM, 0, 0])
                        cmp_kv_hca = pl.slice(hca_cmp_kv, [hca_cmp_block_num, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], [type_index * hca_cmp_block_num, 0, 0, 0])
                        prefill_attention_hca(
                            layer_hidden, hc_attn_fn_layer, hc_attn_scale_layer, hc_attn_base_layer,
                            attn_norm_w_layer, wq_a_layer, wq_b_layer, wq_b_scale_layer, wkv_layer,
                            gamma_cq_layer, gamma_ckv_layer, compressed_freqs_cos, compressed_freqs_sin,
                            hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                            hca_compress_state_hca, hca_compress_state_block_table, kv_cache_layer,
                            ori_slot_mapping_tile, ori_block_table, cmp_kv_hca, hca_cmp_block_table,
                            position_ids_tile, hca_cmp_slot_mapping_tile, hca_state_slot_mapping_tile,
                            attn_sink_layer, wo_a_layer, wo_b_layer, wo_b_scale_layer, x_attn, nt,
                        )
                with pl.scope():
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint='prefill_attention_ready', allow_early_resolve=False) as attention_ready:
                        _attention_ready = pl.read(x_attn, [0, 0, 0])
                    prefill_moe(
                        x_attn, hc_ffn_fn_layer, hc_ffn_scale_layer, hc_ffn_base_layer, norm_w_layer,
                        gate_w_layer, gate_bias_layer, tid2eid_layer, input_ids_tile, routed_w1_layer,
                        routed_w1_scale_layer, routed_w3_layer, routed_w3_scale_layer, routed_w2_layer,
                        routed_w2_scale_layer, shared_w1_layer, shared_w1_scale_layer, shared_w3_layer,
                        shared_w3_scale_layer, shared_w2_layer, shared_w2_scale_layer, pre_hc_hidden_tile,
                        moe_x_mixed, moe_post_ffn, moe_comb_ffn, moe_ffn_out, moe_dense_x, moe_dense_scale,
                        moe_grouped_x, moe_grouped_scale, moe_grouped_y, moe_dense_y, moe_returned_y,
                        count_target, count_signal, x_target, x_signal, scale_target, reverse_target,
                        reverse_signal, attention_ready, layer_index,
                        epoch_base + layer_index + pl.cast(1, pl.INT32), nt,
                    )
                layer_hidden = pre_hc_hidden_tile

            x_head: pl.Tensor[[T, D], pl.BF16] = pl.create_tensor([T, D], dtype=pl.BF16)
            with pl.scope():
                hc_head(pre_hc_hidden_tile, hc_head_fn, hc_head_scale, hc_head_base, x_head)
                rms_norm(x_head, final_norm_w, x_out_tile)

            # Publish each owner's final T valid pre-HC rows in chronological order.
            tail_overlap_start = pl.max(tile_base, owner_tail_start)
            tail_overlap_end = pl.min(tile_base + T, owner_nt)
            tail_overlap_tokens = pl.max(
                tail_overlap_end - tail_overlap_start,
                pl.cast(0, pl.INT32),
            )
            if tail_overlap_tokens > 0:
                tail_source_start = tail_overlap_start - tile_base
                tail_destination_start = tail_overlap_start - owner_tail_start
                for copy_block in pl.spmd(
                    (T // PRE_HC_COPY_TOKEN_TILE) * HC_MULT,
                    name_hint="prefill_pre_hc_tail_copy",
                ):
                    copy_token_block = copy_block // HC_MULT
                    copy_hc_lane = copy_block % HC_MULT
                    copy_token_start = copy_token_block * PRE_HC_COPY_TOKEN_TILE
                    for copy_offset in pl.range(PRE_HC_COPY_TOKEN_TILE):
                        copy_token = copy_token_start + copy_offset
                        if copy_token < tail_overlap_tokens:
                            source_token = tail_source_start + copy_token
                            destination_token = tail_destination_start + copy_token
                            pre_hc_hidden_out[
                                destination_token : destination_token + 1,
                                copy_hc_lane : copy_hc_lane + 1,
                                0:D,
                            ] = pl.slice(
                                pre_hc_hidden_tile,
                                [1, 1, D],
                                [source_token, copy_hc_lane, 0],
                            )

    moe_completion = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_moe_final_anchor"):
        moe_completion[0:1, 0:1, 0:8] = pl.slice(pre_hc_hidden_tile, [1, 1, 8], [0, 0, 0])
    clear_prefill_moe_signals(moe_completion, count_signal, x_signal, reverse_signal)
    return x_out


from prefill_cp import (
    ATTN_TILE_ROWS as CP_FWD_ATTN_TILE_ROWS,
    CMP_META_DIM as CP_FWD_CMP_META_DIM,
    CMP_WINDOW_ROWS as CP_FWD_CMP_WINDOW_ROWS,
    CP_SIZE as CP_FWD_CP_SIZE,
    CP_TAIL_WINDOW_ROWS as CP_FWD_CP_TAIL_WINDOW_ROWS,
    CSA_INNER_STATE_DIM as CP_FWD_CSA_INNER_STATE_DIM,
    CSA_MAIN_OUT_DIM as CP_FWD_CSA_MAIN_OUT_DIM,
    CSA_MAIN_STATE_DIM as CP_FWD_CSA_MAIN_STATE_DIM,
    CSA_MAX_COMPRESS_LEAVES as CP_FWD_CSA_MAX_COMPRESS_LEAVES,
    D as CP_FWD_D,
    HCA_COMPRESS_STATE_DIM as CP_FWD_HCA_COMPRESS_STATE_DIM,
    HC_MULT as CP_FWD_HC_MULT,
    HEAD_DIM as CP_FWD_HEAD_DIM,
    IDX_HEAD_DIM as CP_FWD_IDX_HEAD_DIM,
    IDX_TOPK as CP_FWD_IDX_TOPK,
    LOCAL_PARTS as CP_FWD_LOCAL_PARTS,
    LOCAL_ROWS as CP_FWD_LOCAL_ROWS,
    MAX_SEGMENT_TILES as CP_FWD_MAX_SEGMENT_TILES,
    META_DIM as CP_FWD_META_DIM,
    NUM_SEGMENTS as CP_FWD_NUM_SEGMENTS,
    N_LOCAL as CP_FWD_N_LOCAL,
    N_RANKS as CP_FWD_N_RANKS,
    OVERLAY_ROWS as CP_FWD_OVERLAY_ROWS,
    OVERLAY_SOURCES as CP_FWD_OVERLAY_SOURCES,
    PREFILL_MOE_SCALE_PAD as CP_FWD_PREFILL_MOE_SCALE_PAD,
    PREFILL_MOE_TOTAL_CAP as CP_FWD_PREFILL_MOE_TOTAL_CAP,
    RECORDS_PER_WINDOW as CP_FWD_RECORDS_PER_WINDOW,
    SCALE_TILE_COLS as CP_FWD_SCALE_TILE_COLS,
    STATE_META_DIM as CP_FWD_STATE_META_DIM,
    STATE_RECORDS_PER_WINDOW as CP_FWD_STATE_RECORDS_PER_WINDOW,
    STATE_WINDOW_ROWS as CP_FWD_STATE_WINDOW_ROWS,
    TAIL_ROWS as CP_FWD_TAIL_ROWS,
    WIN as CP_FWD_WIN,
)
# CP needs larger packed root, attention, and sparse-reader workspaces.
PREFILL_RING_HEAP = tuple(
    gib * 1024**3
    for gib in ((2, 2, 2 if CP_FWD_CP_SIZE == 8 else 1, 2) if CP_FWD_CP_SIZE > 1 else (1, 1, 1, 1))
)

from prefill_cp_exchange import (
    CP_REQUEST_CAPACITY as CP_EXCHANGE_CP_REQUEST_CAPACITY,
    CP_REQUEST_HC_DIM as CP_EXCHANGE_CP_REQUEST_HC_DIM,
    CP_REQUEST_INNER_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS,
    CP_REQUEST_MAIN_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS,
    CP_REQUEST_TABLE_COLS as CP_EXCHANGE_CP_REQUEST_TABLE_COLS,
    CP_SIZE as CP_EXCHANGE_CP_SIZE,
    D as CP_EXCHANGE_D,
    HCA_STATE_MAX_BLOCKS as CP_EXCHANGE_HCA_STATE_MAX_BLOCKS,
    LOCAL_ROWS as CP_EXCHANGE_LOCAL_ROWS,
    NUM_SEGMENTS as CP_EXCHANGE_NUM_SEGMENTS,
    PREFILL_CMP_MAX_BLOCKS as CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS as CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS,
    TAIL_ROWS as CP_EXCHANGE_TAIL_ROWS,
)
from prefill_cp import (
    prefill_cp_fwd as cp_forward,
    _prefill_cp_metadata as prepare_cp_metadata,
)
from prefill_cp_exchange import (
    _prefill_cp_request_header as prepare_cp_request,
    _prefill_cp_scatter_request as scatter_cp_request,
    _prefill_cp_publish_hidden as publish_cp_hidden,
)


@pl.jit(auto_scope=False)
def _prefill_dispatch(
    x_hc: pl.Tensor[[FWD_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
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
    ori_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    position_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[FWD_TOKENS_DYN], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    x_out: pl.Out[pl.Tensor[[FWD_TOKENS_DYN, D], pl.BF16]],
    count_target: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    count_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    x_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.INT8],
    x_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    scale_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], pl.FP32],
    reverse_target: pld.DistributedTensor[[PREFILL_MOE_TOTAL_CAP, D], pl.BF16],
    reverse_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    cp_kv_tail_window: pld.DistributedTensor[[CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_HEAD_DIM], pl.BF16],
    cp_hidden_tail_window: pld.DistributedTensor[[CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_D], pl.BF16],
    cp_tail_ready: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_tail_consumed: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_cmp_window: pld.DistributedTensor[[CP_FWD_CMP_WINDOW_ROWS, CP_FWD_HEAD_DIM], pl.BF16],
    cp_cmp_meta_window: pld.DistributedTensor[[CP_FWD_CMP_WINDOW_ROWS, CP_FWD_CMP_META_DIM], pl.INT32],
    cp_state_window: pld.DistributedTensor[[CP_FWD_STATE_WINDOW_ROWS, CP_FWD_HCA_COMPRESS_STATE_DIM], pl.FP32],
    cp_state_meta_window: pld.DistributedTensor[[CP_FWD_CP_SIZE, CP_FWD_STATE_META_DIM], pl.INT32],
    cp_hca_compact_ready: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_hca_compact_consumed: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_main_window: pld.DistributedTensor[[CP_FWD_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_OUT_DIM], pl.BF16],
    cp_idx_window: pld.DistributedTensor[[CP_FWD_RECORDS_PER_WINDOW, CP_FWD_IDX_HEAD_DIM], pl.INT8],
    cp_scale_window: pld.DistributedTensor[[CP_FWD_RECORDS_PER_WINDOW, CP_FWD_SCALE_TILE_COLS], pl.FP16],
    cp_record_window: pld.DistributedTensor[[CP_FWD_RECORDS_PER_WINDOW, CP_FWD_META_DIM], pl.INT32],
    cp_main_state_window: pld.DistributedTensor[[CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_STATE_DIM], pl.FP32],
    cp_main_state_meta_window: pld.DistributedTensor[[CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], pl.INT32],
    cp_inner_state_window: pld.DistributedTensor[[CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_INNER_STATE_DIM], pl.FP32],
    cp_inner_state_meta_window: pld.DistributedTensor[[CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], pl.INT32],
    cp_csa_compact_ready: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_csa_compact_consumed: pld.DistributedTensor[[CP_FWD_CP_SIZE, 1], pl.INT32],
    cp_count_target: pld.DistributedTensor[[CP_FWD_N_RANKS, CP_FWD_N_LOCAL], pl.INT32],
    cp_count_signal: pld.DistributedTensor[[CP_FWD_N_RANKS, 1], pl.INT32],
    cp_prefill_moe_x_target: pld.DistributedTensor[[CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], pl.INT8],
    cp_prefill_moe_x_signal: pld.DistributedTensor[[CP_FWD_N_RANKS, 1], pl.INT32],
    cp_prefill_moe_scale_target: pld.DistributedTensor[[CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_PREFILL_MOE_SCALE_PAD], pl.FP32],
    cp_prefill_moe_reverse_target: pld.DistributedTensor[[CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], pl.BF16],
    cp_prefill_moe_reverse_signal: pld.DistributedTensor[[CP_FWD_N_RANKS, 1], pl.INT32],
    entry_header_window: pld.DistributedTensor[[1, 16], pl.INT32],
    entry_input_window: pld.DistributedTensor[[CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], pl.FP32],
    entry_ids_window: pld.DistributedTensor[[1, CP_EXCHANGE_LOCAL_ROWS * 2], pl.INT32],
    entry_tables_window: pld.DistributedTensor[[7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], pl.INT32],
    entry_ready: pld.DistributedTensor[[CP_EXCHANGE_CP_SIZE, 1], pl.INT32],
    entry_hidden_window: pld.DistributedTensor[[CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], pl.BF16],
    entry_pre_hc_tail_window: pld.DistributedTensor[[CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], pl.FP32],
    entry_complete: pld.DistributedTensor[[CP_EXCHANGE_CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    if CP_FWD_CP_SIZE == 1:
        prefill_fwd(
            x_hc,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_w,
            wq_a,
            wq_b,
            wq_b_scale,
            wkv,
            gamma_cq,
            gamma_ckv,
            kv_cache,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            hca_cmp_kv,
            csa_cmp_kv,
            hca_cmp_wkv,
            hca_cmp_wgate,
            hca_cmp_ape,
            hca_cmp_norm_w,
            hca_compress_state,
            csa_cmp_wkv,
            csa_cmp_wgate,
            csa_cmp_ape,
            csa_cmp_norm_w,
            csa_compress_state,
            csa_hadamard_idx,
            csa_idx_wq_b,
            csa_idx_wq_b_scale,
            csa_weights_proj,
            csa_inner_wkv,
            csa_inner_wgate,
            csa_inner_ape,
            csa_inner_norm_w,
            csa_inner_compress_state,
            idx_kv_cache,
            idx_kv_scale,
            hca_compress_state_block_table,
            csa_compress_state_block_table,
            csa_inner_compress_state_block_table,
            freqs_cos,
            freqs_sin,
            ori_block_table,
            hca_cmp_block_table,
            csa_cmp_block_table,
            idx_block_table,
            ori_slot_mapping,
            position_ids,
            input_ids,
            hca_cmp_slot_mapping,
            hca_state_slot_mapping,
            csa_cmp_slot_mapping,
            csa_idx_slot_mapping,
            csa_state_slot_mapping,
            csa_inner_state_slot_mapping,
            hc_head_fn,
            hc_head_scale,
            hc_head_base,
            final_norm_w,
            pre_hc_hidden_out,
            x_out,
            count_target,
            count_signal,
            x_target,
            x_signal,
            scale_target,
            reverse_target,
            reverse_signal,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            norm_w,
            gate_w,
            gate_bias,
            tid2eid,
            routed_w1,
            routed_w1_scale,
            routed_w3,
            routed_w3_scale,
            routed_w2,
            routed_w2_scale,
            shared_w1,
            shared_w1_scale,
            shared_w3,
            shared_w3_scale,
            shared_w2,
            shared_w2_scale,
            num_tokens_per_owner,
            my_rank,
        )
    else:
        request_rows = pl.tensor.dim(x_hc, 0)
        header = pl.create_tensor([1, 16], dtype=pl.INT32)
        control = pl.create_tensor([1, 16], dtype=pl.INT32)
        input_x_flat = pl.reshape(x_hc, [request_rows, HC_DIM])
        input_ids_row = pl.reshape(input_ids, [1, request_rows])
        ori_block_table_row = pl.reshape(ori_block_table, [1, CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS])
        hca_cmp_block_table_row = pl.reshape(hca_cmp_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        csa_cmp_block_table_row = pl.reshape(csa_cmp_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        idx_block_table_row = pl.reshape(idx_block_table, [1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
        hca_compress_state_block_table_row = pl.reshape(hca_compress_state_block_table, [1, CP_EXCHANGE_HCA_STATE_MAX_BLOCKS])
        csa_compress_state_block_table_row = pl.reshape(csa_compress_state_block_table, [1, CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS])
        csa_inner_compress_state_block_table_row = pl.reshape(csa_inner_compress_state_block_table, [1, CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS])
        local_x_hc = pl.create_tensor([CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
        local_input_ids = pl.create_tensor([1, CP_EXCHANGE_LOCAL_ROWS], dtype=pl.INT64)
        local_ori_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_cmp_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_cmp_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_idx_block_table = pl.create_tensor([1, CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_HCA_STATE_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS], dtype=pl.INT32)
        local_csa_inner_compress_state_block_table = pl.create_tensor([1, CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS], dtype=pl.INT32)
        header = prepare_cp_request(num_tokens_per_owner, position_ids, header, my_rank)
        (control, local_x_hc, local_input_ids, local_ori_block_table, local_hca_cmp_block_table, local_csa_cmp_block_table, local_idx_block_table, local_hca_compress_state_block_table, local_csa_compress_state_block_table, local_csa_inner_compress_state_block_table) = scatter_cp_request(
            header,
            input_x_flat,
            input_ids_row,
            ori_block_table_row,
            hca_cmp_block_table_row,
            csa_cmp_block_table_row,
            idx_block_table_row,
            hca_compress_state_block_table_row,
            csa_compress_state_block_table_row,
            csa_inner_compress_state_block_table_row,
            entry_header_window,
            entry_input_window,
            entry_ids_window,
            entry_tables_window,
            entry_ready,
            control,
            local_x_hc,
            local_input_ids,
            local_ori_block_table,
            local_hca_cmp_block_table,
            local_csa_cmp_block_table,
            local_idx_block_table,
            local_hca_compress_state_block_table,
            local_csa_compress_state_block_table,
            local_csa_inner_compress_state_block_table,
            my_rank,
        )
        mode = pl.read(control, [0, 0])
        if mode == 1:
            cp_ori_block_table = pl.reshape(local_ori_block_table, [CP_EXCHANGE_PREFILL_ORI_MAX_BLOCKS])
            cp_hca_cmp_block_table = pl.reshape(local_hca_cmp_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
            cp_csa_cmp_block_table = pl.reshape(local_csa_cmp_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
            cp_idx_block_table = pl.reshape(local_idx_block_table, [CP_EXCHANGE_PREFILL_CMP_MAX_BLOCKS])
            cp_hca_compress_state_block_table = pl.reshape(local_hca_compress_state_block_table, [CP_EXCHANGE_HCA_STATE_MAX_BLOCKS])
            cp_csa_compress_state_block_table = pl.reshape(local_csa_compress_state_block_table, [CP_EXCHANGE_CP_REQUEST_MAIN_TABLE_COLS])
            cp_csa_inner_compress_state_block_table = pl.reshape(local_csa_inner_compress_state_block_table, [CP_EXCHANGE_CP_REQUEST_INNER_TABLE_COLS])
            segment_starts_t = pl.create_tensor([CP_FWD_NUM_SEGMENTS], dtype=pl.INT32)
            predecessor_segments = pl.create_tensor([CP_FWD_LOCAL_PARTS], dtype=pl.INT32)
            query_position_ids = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            query_token_to_request = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            overlay_position_ids = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_OVERLAY_ROWS], dtype=pl.INT32)
            overlay_token_to_request = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_OVERLAY_ROWS], dtype=pl.INT32)
            overlay_active_lengths = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_OVERLAY_SOURCES], dtype=pl.INT32)
            swa_indices = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_TAIL_ROWS, CP_FWD_WIN], dtype=pl.INT32)
            reverse_index = pl.create_tensor([CP_FWD_NUM_SEGMENTS], dtype=pl.INT32)
            owner_rank_table = pl.create_tensor([CP_FWD_NUM_SEGMENTS], dtype=pl.INT32)
            final_win_seg_src = pl.create_tensor([CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            final_win_row_src = pl.create_tensor([CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            final_slot_mapping = pl.create_tensor([CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            segment_active_lengths = pl.create_tensor([CP_FWD_LOCAL_PARTS], dtype=pl.INT32)
            cache_owner_rank_t = pl.create_tensor([1], dtype=pl.INT32)
            owner_segments_t = pl.create_tensor([CP_FWD_LOCAL_PARTS], dtype=pl.INT32)
            final_segment_t = pl.create_tensor([1], dtype=pl.INT32)
            segment_tail_positions = pl.create_tensor([CP_FWD_NUM_SEGMENTS, CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            snapshot_positions = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_TAIL_ROWS], dtype=pl.INT32)
            snapshot_valid = pl.create_tensor([CP_FWD_LOCAL_PARTS], dtype=pl.INT32)
            owner_part_table = pl.create_tensor([CP_FWD_NUM_SEGMENTS], dtype=pl.INT32)
            cmp_indices = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_TAIL_ROWS, CP_FWD_IDX_TOPK], dtype=pl.INT32)
            segment_lengths_t = pl.create_tensor([CP_FWD_NUM_SEGMENTS], dtype=pl.INT32)
            leaf_positions_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES, CP_FWD_ATTN_TILE_ROWS], dtype=pl.INT32)
            leaf_main_slots_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES, CP_FWD_ATTN_TILE_ROWS], dtype=pl.INT64)
            leaf_idx_slots_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES, CP_FWD_ATTN_TILE_ROWS], dtype=pl.INT64)
            leaf_main_state_slots_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES, CP_FWD_ATTN_TILE_ROWS], dtype=pl.INT64)
            leaf_inner_state_slots_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES, CP_FWD_ATTN_TILE_ROWS], dtype=pl.INT64)
            leaf_num_tokens_input = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_CSA_MAX_COMPRESS_LEAVES], dtype=pl.INT32)
            prepare_cp_metadata(
                control,
                cp_ori_block_table,
                cp_csa_compress_state_block_table,
                cp_csa_inner_compress_state_block_table,
                cp_idx_block_table,
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
                my_rank,
            )
            cp_x_hc = pl.reshape(local_x_hc, [CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_TAIL_ROWS, CP_FWD_HC_MULT, CP_FWD_D])
            cp_input_ids = pl.reshape(local_input_ids, [CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_ATTN_TILE_ROWS])
            cp_pre_hc_hidden_out = pl.create_tensor([CP_FWD_LOCAL_PARTS, CP_FWD_MAX_SEGMENT_TILES, CP_FWD_ATTN_TILE_ROWS, CP_FWD_HC_MULT, CP_FWD_D], dtype=pl.FP32)
            cp_hidden_out = pl.create_tensor([CP_FWD_LOCAL_ROWS, CP_FWD_D], dtype=pl.BF16)
            cp_hidden_out = cp_forward(
                cp_x_hc,
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                attn_norm_w,
                wq_a,
                wq_b,
                wq_b_scale,
                wkv,
                gamma_cq,
                gamma_ckv,
                freqs_cos,
                freqs_sin,
                kv_cache,
                hca_cmp_kv,
                csa_cmp_kv,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
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
                hca_cmp_wkv,
                hca_cmp_wgate,
                hca_cmp_ape,
                hca_cmp_norm_w,
                hca_compress_state,
                cp_hca_compress_state_block_table,
                segment_tail_positions,
                snapshot_positions,
                snapshot_valid,
                owner_part_table,
                cmp_indices,
                csa_cmp_wkv,
                csa_cmp_wgate,
                csa_cmp_ape,
                csa_cmp_norm_w,
                csa_hadamard_idx,
                csa_idx_wq_b,
                csa_idx_wq_b_scale,
                csa_weights_proj,
                csa_inner_wkv,
                csa_inner_wgate,
                csa_inner_ape,
                csa_inner_norm_w,
                csa_compress_state,
                csa_inner_compress_state,
                idx_kv_cache,
                idx_kv_scale,
                cp_csa_compress_state_block_table,
                cp_csa_inner_compress_state_block_table,
                cp_idx_block_table,
                segment_lengths_t,
                leaf_positions_input,
                leaf_main_slots_input,
                leaf_idx_slots_input,
                leaf_main_state_slots_input,
                leaf_inner_state_slots_input,
                leaf_num_tokens_input,
                cp_hca_cmp_block_table,
                cp_csa_cmp_block_table,
                cp_kv_tail_window,
                cp_hidden_tail_window,
                cp_tail_ready,
                cp_tail_consumed,
                cp_cmp_window,
                cp_cmp_meta_window,
                cp_state_window,
                cp_state_meta_window,
                cp_hca_compact_ready,
                cp_hca_compact_consumed,
                cp_main_window,
                cp_idx_window,
                cp_scale_window,
                cp_record_window,
                cp_main_state_window,
                cp_main_state_meta_window,
                cp_inner_state_window,
                cp_inner_state_meta_window,
                cp_csa_compact_ready,
                cp_csa_compact_consumed,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
                norm_w,
                gate_w,
                gate_bias,
                tid2eid,
                cp_input_ids,
                routed_w1,
                routed_w1_scale,
                routed_w3,
                routed_w3_scale,
                routed_w2,
                routed_w2_scale,
                shared_w1,
                shared_w1_scale,
                shared_w3,
                shared_w3_scale,
                shared_w2,
                shared_w2_scale,
                cp_count_target,
                cp_count_signal,
                cp_prefill_moe_x_target,
                cp_prefill_moe_x_signal,
                cp_prefill_moe_scale_target,
                cp_prefill_moe_reverse_target,
                cp_prefill_moe_reverse_signal,
                hc_head_fn,
                hc_head_scale,
                hc_head_base,
                final_norm_w,
                cp_pre_hc_hidden_out,
                cp_hidden_out,
                my_rank,
            )
            cp_pre_hc_flat = pl.reshape(cp_pre_hc_hidden_out, [CP_FWD_LOCAL_ROWS, HC_DIM])
            output_pre_hc_flat = pl.reshape(pre_hc_hidden_out, [T, HC_DIM])
            x_out, output_pre_hc_flat = publish_cp_hidden(
                control,
                cp_hidden_out,
                cp_pre_hc_flat,
                entry_hidden_window,
                entry_pre_hc_tail_window,
                entry_complete,
                x_out,
                output_pre_hc_flat,
                my_rank,
            )
        else:
            prefill_fwd(
                x_hc,
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                attn_norm_w,
                wq_a,
                wq_b,
                wq_b_scale,
                wkv,
                gamma_cq,
                gamma_ckv,
                kv_cache,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                hca_cmp_kv,
                csa_cmp_kv,
                hca_cmp_wkv,
                hca_cmp_wgate,
                hca_cmp_ape,
                hca_cmp_norm_w,
                hca_compress_state,
                csa_cmp_wkv,
                csa_cmp_wgate,
                csa_cmp_ape,
                csa_cmp_norm_w,
                csa_compress_state,
                csa_hadamard_idx,
                csa_idx_wq_b,
                csa_idx_wq_b_scale,
                csa_weights_proj,
                csa_inner_wkv,
                csa_inner_wgate,
                csa_inner_ape,
                csa_inner_norm_w,
                csa_inner_compress_state,
                idx_kv_cache,
                idx_kv_scale,
                hca_compress_state_block_table,
                csa_compress_state_block_table,
                csa_inner_compress_state_block_table,
                freqs_cos,
                freqs_sin,
                ori_block_table,
                hca_cmp_block_table,
                csa_cmp_block_table,
                idx_block_table,
                ori_slot_mapping,
                position_ids,
                input_ids,
                hca_cmp_slot_mapping,
                hca_state_slot_mapping,
                csa_cmp_slot_mapping,
                csa_idx_slot_mapping,
                csa_state_slot_mapping,
                csa_inner_state_slot_mapping,
                hc_head_fn,
                hc_head_scale,
                hc_head_base,
                final_norm_w,
                pre_hc_hidden_out,
                x_out,
                count_target,
                count_signal,
                x_target,
                x_signal,
                scale_target,
                reverse_target,
                reverse_signal,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
                norm_w,
                gate_w,
                gate_bias,
                tid2eid,
                routed_w1,
                routed_w1_scale,
                routed_w3,
                routed_w3_scale,
                routed_w2,
                routed_w2_scale,
                shared_w1,
                shared_w1_scale,
                shared_w3,
                shared_w3_scale,
                shared_w2,
                shared_w2_scale,
                num_tokens_per_owner,
                my_rank,
            )
    return x_out


@pl.jit.host
def l3_prefill_fwd(
    x_hc: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_STATE_BLOCK_SIZE, CSA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, CSA_INNER_COMPRESS_STATE_DIM], pl.FP32]],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, INNER_STATE_MAX_BLOCKS], pl.INT32],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[N_RANKS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[N_RANKS, FWD_TOKENS_DYN], pl.INT64],
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
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, FWD_TOKENS_DYN, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
):

    x_hc.bind_dynamic(1, FWD_TOKENS_DYN)
    hidden_out.bind_dynamic(1, FWD_TOKENS_DYN)
    ori_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    position_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    input_ids.bind_dynamic(1, FWD_TOKENS_DYN)
    hca_cmp_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    hca_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_cmp_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_idx_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(1, FWD_TOKENS_DYN)

    count_target_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    count_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    x_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
    x_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    scale_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
    reverse_target_buf = pld.alloc_window_buffer([PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16)
    reverse_signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    # The LM head owns every window and counter it touches: a peer routes into
    # logits_window while still reading its own hidden_window, and the barrier
    # counters stay independent of the MoE epoch protocol.
    lm_head_hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    lm_head_logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * LM_HEAD_VOCAB * 4)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    cp_kv_tail_window_buf = pld.alloc_window_buffer([CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_HEAD_DIM], dtype=pl.BF16)
    cp_hidden_tail_window_buf = pld.alloc_window_buffer([CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_D], dtype=pl.BF16)
    cp_tail_ready_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_tail_consumed_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_cmp_window_buf = pld.alloc_window_buffer([CP_FWD_CMP_WINDOW_ROWS, CP_FWD_HEAD_DIM], dtype=pl.BF16)
    cp_cmp_meta_window_buf = pld.alloc_window_buffer([CP_FWD_CMP_WINDOW_ROWS, CP_FWD_CMP_META_DIM], dtype=pl.INT32)
    cp_state_window_buf = pld.alloc_window_buffer([CP_FWD_STATE_WINDOW_ROWS, CP_FWD_HCA_COMPRESS_STATE_DIM], dtype=pl.FP32)
    cp_state_meta_window_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
    cp_hca_compact_ready_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_hca_compact_consumed_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_main_window_buf = pld.alloc_window_buffer([CP_FWD_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_OUT_DIM], dtype=pl.BF16)
    cp_idx_window_buf = pld.alloc_window_buffer([CP_FWD_RECORDS_PER_WINDOW, CP_FWD_IDX_HEAD_DIM], dtype=pl.INT8)
    cp_scale_window_buf = pld.alloc_window_buffer([CP_FWD_RECORDS_PER_WINDOW, CP_FWD_SCALE_TILE_COLS], dtype=pl.FP16)
    cp_record_window_buf = pld.alloc_window_buffer([CP_FWD_RECORDS_PER_WINDOW, CP_FWD_META_DIM], dtype=pl.INT32)
    cp_main_state_window_buf = pld.alloc_window_buffer([CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_STATE_DIM], dtype=pl.FP32)
    cp_main_state_meta_window_buf = pld.alloc_window_buffer([CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
    cp_inner_state_window_buf = pld.alloc_window_buffer([CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_INNER_STATE_DIM], dtype=pl.FP32)
    cp_inner_state_meta_window_buf = pld.alloc_window_buffer([CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
    cp_csa_compact_ready_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_csa_compact_consumed_buf = pld.alloc_window_buffer([CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
    cp_count_target_buf = pld.alloc_window_buffer([CP_FWD_N_RANKS, CP_FWD_N_LOCAL], dtype=pl.INT32)
    cp_count_signal_buf = pld.alloc_window_buffer([CP_FWD_N_RANKS, 1], dtype=pl.INT32)
    cp_prefill_moe_x_target_buf = pld.alloc_window_buffer([CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], dtype=pl.INT8)
    cp_prefill_moe_x_signal_buf = pld.alloc_window_buffer([CP_FWD_N_RANKS, 1], dtype=pl.INT32)
    cp_prefill_moe_scale_target_buf = pld.alloc_window_buffer([CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
    cp_prefill_moe_reverse_target_buf = pld.alloc_window_buffer([CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], dtype=pl.BF16)
    cp_prefill_moe_reverse_signal_buf = pld.alloc_window_buffer([CP_FWD_N_RANKS, 1], dtype=pl.INT32)
    entry_header_window_buf = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    entry_input_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
    entry_ids_window_buf = pld.alloc_window_buffer([1, CP_EXCHANGE_LOCAL_ROWS * 2], dtype=pl.INT32)
    entry_tables_window_buf = pld.alloc_window_buffer([7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
    entry_ready_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
    entry_hidden_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], dtype=pl.BF16)
    entry_pre_hc_tail_window_buf = pld.alloc_window_buffer([CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
    entry_complete_buf = pld.alloc_window_buffer([CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        cp_kv_tail_window = pld.window(cp_kv_tail_window_buf, [CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_HEAD_DIM], dtype=pl.BF16)
        cp_hidden_tail_window = pld.window(cp_hidden_tail_window_buf, [CP_FWD_CP_TAIL_WINDOW_ROWS, CP_FWD_D], dtype=pl.BF16)
        cp_tail_ready = pld.window(cp_tail_ready_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_tail_consumed = pld.window(cp_tail_consumed_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_cmp_window = pld.window(cp_cmp_window_buf, [CP_FWD_CMP_WINDOW_ROWS, CP_FWD_HEAD_DIM], dtype=pl.BF16)
        cp_cmp_meta_window = pld.window(cp_cmp_meta_window_buf, [CP_FWD_CMP_WINDOW_ROWS, CP_FWD_CMP_META_DIM], dtype=pl.INT32)
        cp_state_window = pld.window(cp_state_window_buf, [CP_FWD_STATE_WINDOW_ROWS, CP_FWD_HCA_COMPRESS_STATE_DIM], dtype=pl.FP32)
        cp_state_meta_window = pld.window(cp_state_meta_window_buf, [CP_FWD_CP_SIZE, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
        cp_hca_compact_ready = pld.window(cp_hca_compact_ready_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_hca_compact_consumed = pld.window(cp_hca_compact_consumed_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_main_window = pld.window(cp_main_window_buf, [CP_FWD_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_OUT_DIM], dtype=pl.BF16)
        cp_idx_window = pld.window(cp_idx_window_buf, [CP_FWD_RECORDS_PER_WINDOW, CP_FWD_IDX_HEAD_DIM], dtype=pl.INT8)
        cp_scale_window = pld.window(cp_scale_window_buf, [CP_FWD_RECORDS_PER_WINDOW, CP_FWD_SCALE_TILE_COLS], dtype=pl.FP16)
        cp_record_window = pld.window(cp_record_window_buf, [CP_FWD_RECORDS_PER_WINDOW, CP_FWD_META_DIM], dtype=pl.INT32)
        cp_main_state_window = pld.window(cp_main_state_window_buf, [CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_MAIN_STATE_DIM], dtype=pl.FP32)
        cp_main_state_meta_window = pld.window(cp_main_state_meta_window_buf, [CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
        cp_inner_state_window = pld.window(cp_inner_state_window_buf, [CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_CSA_INNER_STATE_DIM], dtype=pl.FP32)
        cp_inner_state_meta_window = pld.window(cp_inner_state_meta_window_buf, [CP_FWD_STATE_RECORDS_PER_WINDOW, CP_FWD_STATE_META_DIM], dtype=pl.INT32)
        cp_csa_compact_ready = pld.window(cp_csa_compact_ready_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_csa_compact_consumed = pld.window(cp_csa_compact_consumed_buf, [CP_FWD_CP_SIZE, 1], dtype=pl.INT32)
        cp_count_target = pld.window(cp_count_target_buf, [CP_FWD_N_RANKS, CP_FWD_N_LOCAL], dtype=pl.INT32)
        cp_count_signal = pld.window(cp_count_signal_buf, [CP_FWD_N_RANKS, 1], dtype=pl.INT32)
        cp_prefill_moe_x_target = pld.window(cp_prefill_moe_x_target_buf, [CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], dtype=pl.INT8)
        cp_prefill_moe_x_signal = pld.window(cp_prefill_moe_x_signal_buf, [CP_FWD_N_RANKS, 1], dtype=pl.INT32)
        cp_prefill_moe_scale_target = pld.window(cp_prefill_moe_scale_target_buf, [CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
        cp_prefill_moe_reverse_target = pld.window(cp_prefill_moe_reverse_target_buf, [CP_FWD_PREFILL_MOE_TOTAL_CAP, CP_FWD_D], dtype=pl.BF16)
        cp_prefill_moe_reverse_signal = pld.window(cp_prefill_moe_reverse_signal_buf, [CP_FWD_N_RANKS, 1], dtype=pl.INT32)
        entry_header_window = pld.window(entry_header_window_buf, [1, 16], dtype=pl.INT32)
        entry_input_window = pld.window(entry_input_window_buf, [CP_EXCHANGE_LOCAL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
        entry_ids_window = pld.window(entry_ids_window_buf, [1, CP_EXCHANGE_LOCAL_ROWS * 2], dtype=pl.INT32)
        entry_tables_window = pld.window(entry_tables_window_buf, [7, CP_EXCHANGE_CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
        entry_ready = pld.window(entry_ready_buf, [CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
        entry_hidden_window = pld.window(entry_hidden_window_buf, [CP_EXCHANGE_CP_REQUEST_CAPACITY, CP_EXCHANGE_D], dtype=pl.BF16)
        entry_pre_hc_tail_window = pld.window(entry_pre_hc_tail_window_buf, [CP_EXCHANGE_NUM_SEGMENTS * CP_EXCHANGE_TAIL_ROWS, CP_EXCHANGE_CP_REQUEST_HC_DIM], dtype=pl.FP32)
        entry_complete = pld.window(entry_complete_buf, [CP_EXCHANGE_CP_SIZE, 1], dtype=pl.INT32)
        count_target = pld.window(count_target_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        count_signal = pld.window(count_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        x_target = pld.window(x_target_buf, [PREFILL_MOE_TOTAL_CAP, D], dtype=pl.INT8)
        x_signal = pld.window(x_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        scale_target = pld.window(scale_target_buf, [PREFILL_MOE_TOTAL_CAP, PREFILL_MOE_SCALE_PAD], dtype=pl.FP32)
        reverse_target = pld.window(reverse_target_buf, [PREFILL_MOE_TOTAL_CAP, D], dtype=pl.BF16)
        reverse_signal = pld.window(reverse_signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        x_hc_rank = x_hc[r]
        hidden_rank = hidden_out[r]
        ori_slot_rank = ori_slot_mapping[r]
        position_ids_rank = position_ids[r]
        input_ids_rank = input_ids[r]
        hca_cmp_slot_rank = hca_cmp_slot_mapping[r]
        hca_state_slot_rank = hca_state_slot_mapping[r]
        csa_cmp_slot_rank = csa_cmp_slot_mapping[r]
        csa_idx_slot_rank = csa_idx_slot_mapping[r]
        csa_state_slot_rank = csa_state_slot_mapping[r]
        csa_inner_state_slot_rank = csa_inner_state_slot_mapping[r]
        _prefill_dispatch(
            x_hc_rank,
            hc_attn_fn[r],
            hc_attn_scale[r],
            hc_attn_base[r],
            attn_norm_w[r],
            wq_a[r],
            wq_b[r],
            wq_b_scale[r],
            wkv[r],
            gamma_cq[r],
            gamma_ckv[r],
            kv_cache[r],
            attn_sink[r],
            wo_a[r],
            wo_b[r],
            wo_b_scale[r],
            hca_cmp_kv[r],
            csa_cmp_kv[r],
            hca_cmp_wkv[r],
            hca_cmp_wgate[r],
            hca_cmp_ape[r],
            hca_cmp_norm_w[r],
            hca_compress_state[r],
            csa_cmp_wkv[r],
            csa_cmp_wgate[r],
            csa_cmp_ape[r],
            csa_cmp_norm_w[r],
            csa_compress_state[r],
            csa_hadamard_idx[r],
            csa_idx_wq_b[r],
            csa_idx_wq_b_scale[r],
            csa_weights_proj[r],
            csa_inner_wkv[r],
            csa_inner_wgate[r],
            csa_inner_ape[r],
            csa_inner_norm_w[r],
            csa_inner_compress_state[r],
            idx_kv_cache[r],
            idx_kv_scale[r],
            hca_compress_state_block_table[r],
            csa_compress_state_block_table[r],
            csa_inner_compress_state_block_table[r],
            freqs_cos[r],
            freqs_sin[r],
            ori_block_table[r],
            hca_cmp_block_table[r],
            csa_cmp_block_table[r],
            idx_block_table[r],
            ori_slot_rank,
            position_ids_rank,
            input_ids_rank,
            hca_cmp_slot_rank,
            hca_state_slot_rank,
            csa_cmp_slot_rank,
            csa_idx_slot_rank,
            csa_state_slot_rank,
            csa_inner_state_slot_rank,
            hc_head_fn[r],
            hc_head_scale[r],
            hc_head_base[r],
            final_norm_w[r],
            pre_hc_hidden_out[r],
            hidden_rank,
            count_target,
            count_signal,
            x_target,
            x_signal,
            scale_target,
            reverse_target,
            reverse_signal,
            hc_ffn_fn[r],
            hc_ffn_scale[r],
            hc_ffn_base[r],
            norm_w[r],
            gate_w[r],
            gate_bias[r],
            tid2eid[r],
            routed_w1[r],
            routed_w1_scale[r],
            routed_w3[r],
            routed_w3_scale[r],
            routed_w2[r],
            routed_w2_scale[r],
            shared_w1[r],
            shared_w1_scale[r],
            shared_w3[r],
            shared_w3_scale[r],
            shared_w2[r],
            shared_w2_scale[r],
            num_tokens_per_owner,
            cp_kv_tail_window,
            cp_hidden_tail_window,
            cp_tail_ready,
            cp_tail_consumed,
            cp_cmp_window,
            cp_cmp_meta_window,
            cp_state_window,
            cp_state_meta_window,
            cp_hca_compact_ready,
            cp_hca_compact_consumed,
            cp_main_window,
            cp_idx_window,
            cp_scale_window,
            cp_record_window,
            cp_main_state_window,
            cp_main_state_meta_window,
            cp_inner_state_window,
            cp_inner_state_meta_window,
            cp_csa_compact_ready,
            cp_csa_compact_consumed,
            cp_count_target,
            cp_count_signal,
            cp_prefill_moe_x_target,
            cp_prefill_moe_x_signal,
            cp_prefill_moe_scale_target,
            cp_prefill_moe_reverse_target,
            cp_prefill_moe_reverse_signal,
            entry_header_window,
            entry_input_window,
            entry_ids_window,
            entry_tables_window,
            entry_ready,
            entry_hidden_window,
            entry_pre_hc_tail_window,
            entry_complete,
            r,
            device=r,
        )

    # Grouped LM head: the N_RANKS DP world is cut into N_RANKS // LM_HEAD_TP_SIZE
    # groups. Every card is both an owner and a TP rank, so the single lm_head
    # dispatch runs on the full world and every peer stays inside its own group.
    for r in pl.range(pld.world_size()):
        hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_test(
            hidden_out[r], lm_head_weight[r], logit_row_indices[r], logits[r],
            hidden_window, hidden_done, logits_window, logits_done,
            r // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE, r % LM_HEAD_TP_SIZE,
            LM_HEAD_COMM_EPOCH, device=r,
        )


# ---------------------------------------------------------------------------
# Fixtures (kernel-only smoke path: no golden).  Stacked weights reuse each
# layer's standalone attention/moe init; routing metadata, slot mappings and
# tid2eid carry meaningful values.
# ---------------------------------------------------------------------------
def _layer_count(name):
    if name in CSA_LAYER_STACKED_NAMES:
        return CSA_NUM_LAYERS
    if name in HCA_LAYER_STACKED_NAMES:
        return HCA_NUM_LAYERS
    if name in FWD_LAYER_STACKED_NAMES:
        return FWD_NUM_LAYERS
    return 1


def _make_stacked_spec(name, base_specs, cache_block_nums=None):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    count = _layer_count(name)
    packed_shape = [spec.shape[0], count * spec.shape[1], *spec.shape[2:]]

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
        base_init = spec.init_value
        return torch.cat([base_init() for _ in range(count)], dim=1)

    # Caches the kernel writes in place (kv_cache) are outputs read back for
    # validation; every other stacked tensor is a plain input (weight or read-only
    # cache).
    return TensorSpec(
        name, packed_shape, spec.dtype, init_value=init_value,
    )


def _make_shared_spec(name, base_specs, start_pos, cache_block_nums=None):
    import torch
    from golden import TensorSpec

    spec = base_specs[name]
    pos = torch.arange(start_pos, start_pos + T, dtype=torch.int64)

    def ranked(single):
        return single.unsqueeze(0).expand(N_RANKS, *single.shape).contiguous()

    def init_value():
        if name == "position_ids":
            return ranked(pos.to(torch.int32))
        if name == "input_ids":
            return ranked((torch.arange(T, dtype=torch.int64) % VOCAB))
        if name == "ori_slot_mapping":
            return ranked(pos.to(torch.int64))
        if name in ("hca_state_slot_mapping", "csa_state_slot_mapping", "csa_inner_state_slot_mapping"):
            # State mappings carry the physical row, not the logical position.
            block_size, max_blocks, state_name = {
                "hca_state_slot_mapping": (
                    HCA_STATE_BLOCK_SIZE, HCA_STATE_MAX_BLOCKS, "hca_compress_state"),
                "csa_state_slot_mapping": (
                    CSA_STATE_BLOCK_SIZE, CSA_STATE_MAX_BLOCKS, "csa_compress_state"),
                "csa_inner_state_slot_mapping": (
                    INNER_STATE_BLOCK_SIZE, INNER_STATE_MAX_BLOCKS, "csa_inner_compress_state"),
            }[name]
            physical_blocks = cache_block_nums[state_name]
            block = pos // block_size
            row = (block % physical_blocks) * block_size + pos % block_size
            addressable = (pos >= 0) & (pos < MAX_SEQ_LEN) & (block < max_blocks)
            return ranked(torch.where(addressable, row, torch.full_like(row, -1)))
        if name == "hca_cmp_slot_mapping":
            out = torch.full((T,), -1, dtype=torch.int64)
            mask = ((pos + 1) % HCA_COMPRESS_RATIO) == 0
            out[mask] = ((pos[mask] + 1) // HCA_COMPRESS_RATIO) - 1
            return ranked(out)
        if name in ("csa_cmp_slot_mapping", "csa_idx_slot_mapping"):
            out = torch.full((T,), -1, dtype=torch.int64)
            mask = ((pos + 1) % CSA_COMPRESS_RATIO) == 0
            out[mask] = ((pos[mask] + 1) // CSA_COMPRESS_RATIO) - 1
            return ranked(out)
        if name in (
            "ori_block_table",
            "hca_cmp_block_table",
            "csa_cmp_block_table",
            "idx_block_table",
        ):
            physical_pages = {
                "ori_block_table": cache_block_nums["kv_cache"],
                "hca_cmp_block_table": cache_block_nums["hca_cmp_kv"],
                "csa_cmp_block_table": cache_block_nums["csa_cmp_kv"],
                "idx_block_table": cache_block_nums["idx_kv_cache"],
            }[name]
            out = torch.arange(spec.shape[-1], dtype=spec.dtype) % physical_pages
            return ranked(out)
        if name in ("hca_compress_state_block_table", "csa_compress_state_block_table",
                    "csa_inner_compress_state_block_table"):
            state_name = {
                "hca_compress_state_block_table": "hca_compress_state",
                "csa_compress_state_block_table": "csa_compress_state",
                "csa_inner_compress_state_block_table": "csa_inner_compress_state",
            }[name]
            out = torch.arange(spec.shape[-1], dtype=spec.dtype) % cache_block_nums[state_name]
            return ranked(out)
        # Any remaining shared metadata: smoke zeros.
        return torch.zeros(list(spec.shape), dtype=spec.dtype)

    return TensorSpec(name, list(spec.shape), spec.dtype, init_value=init_value)


def _make_hc_head_spec(name):
    import torch
    from golden import TensorSpec

    if name == "hc_head_fn":
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT, HC_DIM],
            torch.float32,
            init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519,
        )
    if name == "hc_head_scale":
        return TensorSpec(
            name,
            [N_RANKS, 1],
            torch.float32,
            init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32),
        )
    if name == "hc_head_base":
        base = [5.9166, -3.6223, -2.9324, -3.3124]
        return TensorSpec(
            name,
            [N_RANKS, HC_MULT],
            torch.float32,
            init_value=lambda: torch.tensor(base, dtype=torch.float32).view(1, HC_MULT).expand(N_RANKS, -1).contiguous(),
        )
    raise ValueError(f"unclassified hc_head spec: {name}")


def _make_final_norm_spec(name):
    import torch
    from golden import TensorSpec

    if name == "final_norm_w":
        return TensorSpec(
            name,
            [N_RANKS, D],
            torch.bfloat16,
            init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16),
        )
    raise ValueError(f"unclassified final norm spec: {name}")


# Canonical host-tensor order for a single unified prefill layer.
HOST_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "freqs_cos",
    "freqs_sin",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "hca_compress_state_block_table",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_compress_state_block_table",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "csa_inner_compress_state_block_table",
    "kv_cache",
    "ori_block_table",
    "ori_slot_mapping",
    "hca_cmp_kv",
    "csa_cmp_kv",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "position_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "input_ids",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "x_next",
)


def _spec_value(spec, torch):
    init_value = getattr(spec, "init_value", None)
    if callable(init_value):
        return init_value()
    if init_value is not None:
        return init_value.clone() if hasattr(init_value, "clone") else init_value
    return torch.zeros(spec.shape, dtype=spec.dtype)


def _ranked_init(spec, n_ranks, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        return torch.stack(values, dim=0).contiguous()

    return init


def _ranked_x_hc_init(spec, n_ranks, active_tokens, torch):
    def init():
        values = [_spec_value(spec, torch) for _ in range(n_ranks)]
        stacked = torch.stack(values, dim=0).contiguous()
        active = min(active_tokens, stacked.shape[1])
        if active < stacked.shape[1]:
            inactive = torch.randn(stacked[:, active:].shape, dtype=torch.float32).to(stacked.dtype)
            stacked[:, active:] = inactive / 10.0
        return stacked

    return init


def _attention_kind_for_layer(layer_id):
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 128:
        return "hca"
    if ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeek V4 attention compress ratio {ratio} at layer {layer_id}")


def build_single_layer_tensor_specs(start_pos=START_POS, num_tokens=T, layer_id=2):
    """Per-layer single-rank tensor specs: the base shapes/dtypes/inits that
    build_tensor_specs restacks across the forward layers."""
    import torch
    from golden import ScalarSpec, TensorSpec

    def kind_specs(build_fn):
        return {s.name: s for s in build_fn(start_pos=start_pos, num_tokens=num_tokens) if isinstance(s, TensorSpec)}

    swa = kind_specs(build_swa_attention_tensor_specs)
    hca = kind_specs(build_hca_attention_tensor_specs)
    csa = kind_specs(build_csa_attention_tensor_specs)
    active_kind = _attention_kind_for_layer(layer_id)
    active = {"swa": swa, "hca": hca, "csa": csa}[active_kind]
    active_tokens = num_tokens

    # (layer_name, source_spec). Shared state is taken from the active kind (its
    # init is what the active attention + its golden both consume). The hca_/csa_
    # compressor + indexer params are namespaced from their own kind; compressed
    # KV specs prefer the active attention kind and fall back to CSA for SWA.
    attention_specs = [
        ("x_hc", active["x_hc"]),
        ("hc_attn_fn", active["hc_attn_fn"]),
        ("hc_attn_scale", active["hc_attn_scale"]),
        ("hc_attn_base", active["hc_attn_base"]),
        ("attn_norm_w", active["attn_norm_w"]),
        ("wq_a", active["wq_a"]),
        ("wq_b", active["wq_b"]),
        ("wq_b_scale", active["wq_b_scale"]),
        ("wkv", active["wkv"]),
        ("gamma_cq", active["gamma_cq"]),
        ("gamma_ckv", active["gamma_ckv"]),
        (
            "freqs_cos",
            TensorSpec(
                "freqs_cos",
                [2, *swa["freqs_cos"].shape],
                swa["freqs_cos"].dtype,
                init_value=lambda: torch.stack(
                    (_spec_value(swa["freqs_cos"], torch), _spec_value(csa["freqs_cos"], torch)),
                    dim=0,
                ),
            ),
        ),
        (
            "freqs_sin",
            TensorSpec(
                "freqs_sin",
                [2, *swa["freqs_sin"].shape],
                swa["freqs_sin"].dtype,
                init_value=lambda: torch.stack(
                    (_spec_value(swa["freqs_sin"], torch), _spec_value(csa["freqs_sin"], torch)),
                    dim=0,
                ),
            ),
        ),
        ("hca_cmp_wkv", hca["cmp_wkv"]),
        ("hca_cmp_wgate", hca["cmp_wgate"]),
        ("hca_cmp_ape", hca["cmp_ape"]),
        ("hca_cmp_norm_w", hca["cmp_norm_w"]),
        ("hca_compress_state", hca["compress_state"]),
        ("hca_compress_state_block_table", hca["compress_state_block_table"]),
        ("csa_cmp_wkv", csa["cmp_wkv"]),
        ("csa_cmp_wgate", csa["cmp_wgate"]),
        ("csa_cmp_ape", csa["cmp_ape"]),
        ("csa_cmp_norm_w", csa["cmp_norm_w"]),
        ("csa_compress_state", csa["compress_state"]),
        ("csa_compress_state_block_table", csa["compress_state_block_table"]),
        ("csa_hadamard_idx", csa["hadamard_idx"]),
        ("csa_idx_wq_b", csa["idx_wq_b"]),
        ("csa_idx_wq_b_scale", csa["idx_wq_b_scale"]),
        ("csa_weights_proj", csa["idx_weights_proj"]),
        ("csa_inner_wkv", csa["inner_wkv"]),
        ("csa_inner_wgate", csa["inner_wgate"]),
        ("csa_inner_ape", csa["inner_ape"]),
        ("csa_inner_norm_w", csa["inner_norm_w"]),
        ("csa_inner_compress_state", csa["inner_compress_state"]),
        ("csa_inner_compress_state_block_table", csa["inner_compress_state_block_table"]),
        ("kv_cache", active["kv_cache"]),
        ("ori_block_table", active.get("ori_block_table", swa.get("block_table"))),
        ("ori_slot_mapping", active["ori_slot_mapping"]),
        ("hca_cmp_kv", hca["cmp_kv"]),
        ("csa_cmp_kv", csa["cmp_kv"]),
        ("hca_cmp_block_table", hca["cmp_block_table"]),
        ("csa_cmp_block_table", csa["cmp_block_table"]),
        ("idx_kv_cache", csa["idx_kv_cache"]),
        ("idx_kv_scale", csa["idx_kv_scale"]),
        ("idx_block_table", csa["idx_block_table"]),
        ("position_ids", active["position_ids"]),
        ("hca_cmp_slot_mapping", hca["cmp_slot_mapping"]),
        ("hca_state_slot_mapping", hca["state_slot_mapping"]),
        ("csa_cmp_slot_mapping", csa["cmp_slot_mapping"]),
        ("csa_idx_slot_mapping", csa["idx_slot_mapping"]),
        ("csa_state_slot_mapping", csa["state_slot_mapping"]),
        ("csa_inner_state_slot_mapping", csa["inner_state_slot_mapping"]),
        ("attn_sink", active["attn_sink"]),
        ("wo_a", active["wo_a"]),
        ("wo_b", active["wo_b"]),
        ("wo_b_scale", active["wo_b_scale"]),
    ]

    tensor_specs = [
        TensorSpec(
            name,
            [N_RANKS, *src.shape],
            src.dtype,
            init_value=(_ranked_x_hc_init(src, N_RANKS, active_tokens, torch) if name == "x_hc"
                        else _ranked_init(src, N_RANKS, torch)),
        )
        for name, src in attention_specs
    ]

    for spec in build_moe_tensor_specs(layer_id=layer_id, token_capacity=T):
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
            def init_input_ids(spec=spec):
                _, tokens = spec.shape
                active = min(active_tokens, tokens)
                rows = []
                for rank in range(N_RANKS):
                    row = torch.roll(torch.arange(tokens, dtype=spec.dtype), shifts=rank)
                    if layer_id >= 3 and active < tokens:
                        row[active:] = -1
                    rows.append(row)
                return torch.stack(rows, dim=0).contiguous()

            tensor_specs.append(TensorSpec(spec.name, spec.shape, spec.dtype, init_value=init_input_ids))
        else:
            tensor_specs.append(spec)

    tensor_specs.append(TensorSpec("x_next", [N_RANKS, T, HC_MULT, D], torch.float32))
    tensor_by_name = {spec.name: spec for spec in tensor_specs}
    missing = [name for name in HOST_TENSOR_ORDER if name not in tensor_by_name]
    if missing:
        raise ValueError(f"missing unified prefill layer tensor specs: {missing}")
    return [tensor_by_name[name] for name in HOST_TENSOR_ORDER] + [
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        ScalarSpec("layer_id", torch.int32, layer_id),
    ]


def build_tensor_specs(
    start_pos=0,
    num_tokens=T,
    num_tiles=1,
    ori_block_num=CSA_ORI_BLOCK_NUM,
    cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=PREFILL_IDX_BLOCK_NUM,
    hca_state_block_num=HCA_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_STATE_BLOCK_NUM,
    inner_state_block_num=INNER_STATE_BLOCK_NUM,
    active_ranks=N_RANKS,
):
    import torch
    from golden import TensorSpec

    def init_lm_head_weight():
        shards = (torch.randn(LM_HEAD_TP_SIZE, VOCAB_PER_TP, D) / D ** 0.5).to(torch.bfloat16)
        return torch.stack([shards[r % LM_HEAD_TP_SIZE] for r in range(N_RANKS)], dim=0)

    # Serving batches one request per rank, so idle ranks carry zero tokens and
    # an all -1 index row. active_ranks reproduces that skew.
    def init_logit_row_indices():
        indices = torch.full((N_RANKS, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
        indices[:active_ranks, 0] = max(min(num_tokens, num_tiles * T), 1) - 1
        return indices

    def init_num_tokens_per_owner():
        counts = torch.zeros(N_RANKS, dtype=torch.int32)
        counts[:active_ranks] = num_tokens
        return counts

    first_tile_tokens = max(1, min(num_tokens, T))
    base_specs = {
        spec.name: spec
        for spec in build_single_layer_tensor_specs(start_pos=start_pos, num_tokens=first_tile_tokens, layer_id=0)
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
        "ori_slot_mapping", "position_ids", "input_ids",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
        "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
        "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
        "gate_w", "gate_bias", "tid2eid",
        "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
        "routed_w2", "routed_w2_scale",
        "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
        "shared_w2", "shared_w2_scale",
        "hc_head_fn", "hc_head_scale", "hc_head_base",
        "final_norm_w",
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
    TILED_NAMES = {
        "ori_slot_mapping", "position_ids", "input_ids",
        "hca_cmp_slot_mapping", "hca_state_slot_mapping",
        "csa_cmp_slot_mapping", "csa_idx_slot_mapping",
        "csa_state_slot_mapping", "csa_inner_state_slot_mapping",
    }

    def make_tiled_shared_spec(name):
        per_tile = [
            _make_shared_spec(name, base_specs, start_pos + tile * T, cache_block_nums)
            for tile in range(num_tiles)
        ]
        head = per_tile[0]
        if num_tiles == 1:
            return head
        shape = list(head.shape)
        shape[1] = num_tiles * T

        def init_joined(parts=per_tile):
            return torch.cat(
                [p.init_value() if callable(p.init_value) else p.init_value for p in parts],
                dim=1,
            )

        return TensorSpec(name, shape, head.dtype, init_value=init_joined)

    specs = []
    for name in ordered_names:
        if name == "x_hc":
            base = base_specs[name]
            x_hc_shape = list(base.shape)
            x_hc_shape[1] = num_tiles * T

            def init_x_hc(shape=x_hc_shape, dtype=base.dtype):
                return (torch.randn(shape) * 0.05).to(dtype)

            specs.append(TensorSpec(name, x_hc_shape, base.dtype, init_value=init_x_hc))
        elif name in TILED_NAMES:
            specs.append(make_tiled_shared_spec(name))
        elif name in SHARED_NAMES:
            specs.append(_make_shared_spec(name, base_specs, start_pos, cache_block_nums))
        elif name in HC_HEAD_NAMES:
            specs.append(_make_hc_head_spec(name))
        elif name in FINAL_NORM_NAMES:
            specs.append(_make_final_norm_spec(name))
        else:
            specs.append(_make_stacked_spec(name, base_specs, cache_block_nums))

    # Shard the static weight parameters per rank and keep them device-resident
    # (child_memory): each shard uploaded once to its card and reused across
    # dispatches, skipping per-dispatch H2D/D2H. RESIDENT_WEIGHT_NAMES are static
    # weights; RESIDENT_CACHE_NAMES are the KV/state caches (the written kv_cache
    # is also an InOut, read back at the end via RESIDENT_CACHE_OUTPUT_NAMES).
    for spec in specs:
        if spec.name in RESIDENT_WEIGHT_NAMES or spec.name in RESIDENT_CACHE_NAMES:
            spec.resident = "stacked"

    specs.append(TensorSpec("pre_hc_hidden_out", [N_RANKS, T, HC_MULT, D], torch.float32))
    specs.append(TensorSpec(
        "lm_head_weight",
        [N_RANKS, VOCAB_PER_TP, D],
        torch.bfloat16,
        init_value=init_lm_head_weight,
        resident="stacked",
    ))
    specs.append(TensorSpec("hidden_out", [N_RANKS, num_tiles * T, D], torch.bfloat16))
    specs.append(TensorSpec(
        "logits",
        [N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB],
        torch.float32,
    ))
    specs.append(TensorSpec(
        "num_tokens_per_owner",
        [N_RANKS],
        torch.int32,
        init_value=init_num_tokens_per_owner,
    ))
    specs.append(TensorSpec(
        "logit_row_indices",
        [N_RANKS, MAX_LOGIT_ROWS],
        torch.int32,
        init_value=init_logit_row_indices,
    ))
    return specs


def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 Flash packed-prefill forward driver.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
                        help="EP world size / rank count (parsed at import by moe).")
    parser.add_argument("--cp", type=int, default=CP_FWD_CP_SIZE, choices=sorted({1, N_RANKS}),
                        help="Context parallel group size: 1 or the full EP world.")
    parser.add_argument("--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[2, 4, 8, 16],
                        help="LM-head TP world size; must be <= --ep.")
    parser.add_argument("-d", "--device", type=str, default=",".join(str(i) for i in range(N_RANKS)),
                        help=f"comma-separated device ids; need at least {N_RANKS}")
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--active-ranks", type=int, default=N_RANKS,
                        help="Ranks carrying tokens; the rest stay idle as in single-request serving.")
    parser.add_argument("--num-tokens", type=int, default=T // 2,
                        help=f"Active token rows for MoE routing/combine; default is T // 2={T // 2}.")
    parser.add_argument("--ori-block-num", type=int, default=CSA_ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=PREFILL_IDX_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=INNER_STATE_BLOCK_NUM)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)

    parser.add_argument("--num-tiles", type=int, default=1,
                        help="fixed-T tiles in one submitted request; 1 reproduces the packed graph.")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    device_ids = [int(d) for d in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"
    assert args.tp <= args.ep, f"--tp must be <= --ep, got tp={args.tp}, ep={args.ep}"
    assert args.ep % args.tp == 0, (
        f"grouped LM head needs --ep % --tp == 0, got ep={args.ep}, tp={args.tp}"
    )
    assert LM_HEAD_TP_SIZE == args.tp, (
        f"import-time LM_HEAD_TP_SIZE must match --tp, got {LM_HEAD_TP_SIZE} vs {args.tp}"
    )
    assert N_RANKS == args.ep, f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}"

    specs = build_tensor_specs(
        start_pos=args.start_pos,
        num_tokens=args.num_tokens,
        ori_block_num=args.ori_block_num,
        cmp_block_num=args.cmp_block_num,
        idx_block_num=args.idx_block_num,
        hca_state_block_num=args.hca_state_block_num,
        csa_state_block_num=args.csa_state_block_num,
        inner_state_block_num=args.inner_state_block_num,
        active_ranks=args.active_ranks,
        num_tiles=args.num_tiles,
    )

    result = run(
        fn=l3_prefill_fwd,
        specs=specs,
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(device_ids=device_ids[:N_RANKS], num_sub_workers=0),
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

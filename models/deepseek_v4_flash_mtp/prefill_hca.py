# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill HCA (ratio-128) attention over one contiguous run of <=T tokens."""
# ci: devices=2

import functools

import sys

# Standalone CI passes its borrowed device list without a --cp argument.
# Resolve fixture topology before importing modules that freeze CP shapes.
if __name__ == "__main__":
    import argparse

    fixture_parser = argparse.ArgumentParser(add_help=False)
    fixture_parser.add_argument("-d", "--device", default="0")
    fixture_parser.add_argument("--cp", type=int, default=None)
    fixture_args, _ = fixture_parser.parse_known_args()
    _run_cp_fixture = fixture_args.cp is not None or "," in fixture_args.device
    if fixture_args.cp is None and _run_cp_fixture:
        sys.argv += ["--cp", str(len(fixture_args.device.split(",")))]

import argparse
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from prefill_compressor_ratio128 import (
    COMPRESS_STATE_DIM,
    build_tensor_specs as build_compressor_tensor_specs,
)
from prefill_cp_exchange import (
    CP_CMP_BLOCK_NUM_DYN,
    HCA_STATE_BLOCKS_DYN,
    CMP_META_DIM,
    CMP_ROWS_PER_RANK,
    CMP_ROWS_PER_SEGMENT,
    CMP_WINDOW_ROWS,
    STATE_META_DIM,
    STATE_WINDOW_ROWS,
    _prefill_cp_hidden_tail_exchange_wave,
    _prefill_cp_hca_compact_exchange_commit_wave,
)
from prefill_cp_zigzag import (
    CP_CHOICES,
    CP_PREFILL_CMP_BLOCK_NUM,
    CP_SIZE,
    CP_TAIL_WINDOW_ROWS,
    EPOCHS,
    MAX_SEGMENT_TILES,
    NUM_SEGMENTS,
    ROW_TILE,
    TAIL_ROWS,
    cp_final_window_sources,
    cp_owner_part,
    cp_owner_rank,
    cp_owner_tables,
    cp_reverse_index,
    cp_segment_layout,
)
from hc_post import hc_post_prefill
from prefill_sparse_attn import (
    HCA_MAX_COMPRESSED_ROWS,
    PREFILL_SPARSE_PAD,
    hca_attn,
    build_tensor_specs as build_sparse_attn_tensor_specs,
)
from qkv_proj_rope import build_tensor_specs as build_qkv_tensor_specs, rope_prepare
from utils import build_rope_tables

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    FLASH as M,
    HCA_STATE_PHYSICAL_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from hc_post import golden_hc_post_prefill
from hc_pre import golden_hc_pre
from prefill_compressor_ratio128 import (
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from qkv_proj_rope import golden_qkv_proj_rope, materialize_rope_rows, prefill_attention_prolog, kv_proj_rope
from rmsnorm import golden_rms_norm
from prefill_sparse_attn import (
    PREFILL_ATTN_TILE,
    SPARSE_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
    golden_prefill_sparse_attn,
    prefill_physical_attention,
)


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_CMP_BLOCK_NUM_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")

# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO
MAIN_OUT_DIM = HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0

# paged KV cache
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
SPARSE_ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
SPARSE_CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
HCA_ORI_BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
HCA_CMP_BLOCK_NUM = SPARSE_CMP_BLOCK_NUM

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
# HCA has no indexer: the compressed tail is every slot the cache holds, so the
# shared prefill pruning width must cover the whole cache, not a top-k budget.
assert MAX_SEQ_LEN // COMPRESS_RATIO <= PREFILL_MAX_COMPRESSED, (
    f"prefill HCA compressed tail ({PREFILL_MAX_COMPRESSED} slots) must cover "
    f"MAX_SEQ_LEN={MAX_SEQ_LEN} ({MAX_SEQ_LEN // COMPRESS_RATIO} slots)")


@pl.jit.inline
def prefill_attention_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rms_tid = prefill_attention_prolog(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos_t, rope_sin_t,
        x_normed, post, comb, cos_il, sin_signed, swap_idx, q, qr, qr_scale,
    )
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    kv_proj_rope(x_normed, wkv, gamma_ckv, cos_il, sin_signed, swap_idx, kv, late_dep)


    ori_block_num = pl.tensor.dim(kv_cache, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_write"):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    prefill_compressor_ratio128(
        x_normed, compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )

    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    cmp_indices = pl.create_tensor([T, IDX_TOPK], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor(
        [T, VALID_BLOCK_MASK_COLS], dtype=pl.INT32
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_sparse_indices"):
        for idx_t in pl.range(T):
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            cmp_row = pl.full([1, IDX_TOPK], dtype=pl.INT32, value=-1)
            mask_row = pl.full(
                [1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0
            )
            if idx_t < num_tokens:
                abs_pos = pl.read(position_ids, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(ori_block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(swa_row, [0, win_col], row)
                            if win_col < SPARSE_BIAS_COLS:
                                pl.write(
                                    mask_row,
                                    [0, win_col // PREFILL_ATTN_TILE],
                                    pl.cast(1, pl.INT32),
                                )
                visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                for cmp_col in pl.range(IDX_TOPK):
                    cmp_col_i32 = pl.cast(cmp_col, pl.INT32)
                    if cmp_col_i32 < visible_cmp:
                        if cmp_col_i32 < pl.cast(
                            SPARSE_CMP_MAX_BLOCKS * CMP_STORAGE_BLOCK_SIZE, pl.INT32
                        ):
                            pl.write(cmp_row, [0, cmp_col], cmp_col_i32)
                            sparse_col = WIN + cmp_col
                            if sparse_col < SPARSE_BIAS_COLS:
                                pl.write(
                                    mask_row,
                                    [0, sparse_col // PREFILL_ATTN_TILE],
                                    pl.cast(1, pl.INT32),
                                )
            swa_indices = pl.assemble(swa_indices, swa_row, [idx_t, 0])
            cmp_indices = pl.assemble(cmp_indices, cmp_row, [idx_t, 0])
            valid_block_mask = pl.assemble(
                valid_block_mask, mask_row, [idx_t, 0]
            )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_sparse_sources_ready") as sources_ready:
        _q_ready = pl.read(q, [0, 0, 0])
        _raw_ready = pl.read(kv_cache, [0, 0, 0, 0])
        _compressed_ready = pl.read(cmp_kv, [0, 0, 0, 0])
        _raw_indices_ready = pl.read(swa_indices, [0, 0])
        _compressed_indices_ready = pl.read(cmp_indices, [0, 0])
        _mask_ready = pl.read(valid_block_mask, [0, 0])
    prefill_physical_attention(
        q, kv_cache, swa_indices, cmp_kv, cmp_block_table, cmp_indices, valid_block_mask,
        attn_sink, rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale,
        x_hc, post, comb, x_out, num_tokens, sources_ready, sources_ready,
    )
    return x_out


@pl.jit
def prefill_attention_hca_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[
        [STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32
    ],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_slot_mapping, ori_block_table,
        cmp_kv, cmp_block_table,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, num_tokens,
    )
    return x_out


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_prefill_attention_hca(tensors):
    import torch

    from utils import cache_row_from_table

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_hc_flat = x_hc_rect.view(T, HC_MULT, D)
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_flat,
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed.view(T, D),
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    ori_kv = tensors["kv_cache"]
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]

    cmp_kv = tensors["cmp_kv"]
    golden_prefill_compressor_ratio128({
        "x": x_normed.view(T, D),
        "compress_state": tensors["compress_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "cmp_kv": cmp_kv,
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })

    def build_sparse_metadata():
        swa_idx = torch.full((T, WIN), -1, dtype=torch.int32)
        cmp_idx = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        ori_table = tensors["ori_block_table"]
        cmp_cap = SPARSE_CMP_MAX_BLOCKS * CMP_STORAGE_BLOCK_SIZE
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(ori_table, key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, IDX_TOPK, cmp_cap)
            if visible_cmp > 0:
                cmp_idx[t, :visible_cmp] = torch.arange(visible_cmp, dtype=torch.int32)
        return swa_idx, cmp_idx

    swa_indices, cmp_indices = build_sparse_metadata()
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": ori_kv,
        "swa_indices": swa_indices,
        "cmp_kv": cmp_kv,
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_storage_block_size": CMP_STORAGE_BLOCK_SIZE,
        "cmp_indices": cmp_indices,
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.float32)
    golden_hc_post_prefill({
        "x": attn_out.view(T, D),
        "residual": x_hc_flat,
        "post": post,
        "comb": comb,
        "y": y.view(T, HC_MULT, D),
        "num_tokens": tensors["num_tokens"],
    })
    tensors["x_out"][:] = y.view(T, HC_MULT, D)


@functools.lru_cache(maxsize=None)
def _state_block_table(max_blocks, physical_blocks):
    """Constant scrambled state block table [max_blocks]."""
    import torch
    blocks = torch.arange(max_blocks, dtype=torch.int32)
    return (blocks * 17 + 3) % physical_blocks


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = T,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables, cache_row_from_table, quant_w_per_channel

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    # Single-request geometry: q_len = num_tokens (active prefix), context_len =
    # start_pos (absolute position base, a multiple of S=WIN under chunked prefill).
    context_len = start_pos
    q_len = num_tokens
    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if context_len < 0:
        raise ValueError(f"context length must be non-negative, got {context_len}")
    max_position = context_len + q_len - 1
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")

    def token_meta():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        # Padding rows keep their arange default; they are inactive.
        local_pos = torch.zeros(T, dtype=torch.int32)
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(q_len):
            local_pos[local_s] = local_s
            pos[local_s] = context_len + local_s
        return local_pos, pos

    def cmp_write_records():
        records = []
        for local_s in range(q_len):
            abs_len = context_len + local_s + 1
            if abs_len >= COMPRESS_RATIO and abs_len % COMPRESS_RATIO == 0:
                token_id = local_s
                cmp_slot = abs_len // COMPRESS_RATIO - 1
                records.append((token_id, cmp_slot))
        return records


    def init_x_hc():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-9 (HCA, ratio-128) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    # Mirrors decode_hca.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495
    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])
    def init_hc_attn_base():
        return torch.tensor([
            -3.3004, 2.5553, -2.2787, -3.4925,
            -3.8197, -3.4161, -2.7144, -2.9181,
            2.362, -2.4746, -2.1352, -3.2216,
            -4.474, 2.2488, -2.1053, -3.1675,
            -2.8362, -1.9042, 2.0432, -3.062,
            -2.7902, -3.0908, -3.002, 3.1161,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D ** -0.5
    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA ** -0.5
    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D ** -0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    # Quant-faithful HCA (ratio-128) main compressor fixtures (mean l7/l9 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_hca / decode_compressor_ratio128.
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0246
    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0316
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0340
    def init_cmp_norm_w():
        return 0.1001 + torch.randn(HEAD_DIM,) * 0.0549
    state_table = _state_block_table(HCA_STATE_MAX_BLOCKS, HCA_STATE_PHYSICAL_BLOCKS)
    def init_compress_state_block_table():
        return state_table.clone()
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(state_table[block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
        flat = state.view(-1, MAIN_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, context_len - COMPRESS_RATIO), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_COMPRESS_STATE_DIM,) - 0.5) * 0.05
        return state
    def init_kv_cache():
        cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()
        if context_len > 0:
            prefix_start = max(0, context_len - WIN)
            prefix = ((torch.rand(context_len, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for pos_i in range(prefix_start, context_len):
                row = cache_row_from_table(table, pos_i)
                if row >= 0:
                    cache_flat[row] = prefix[pos_i]
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        local_pos, _ = token_meta()
        table = init_ori_block_table()
        for t in range(num_tokens):
            logical_pos = context_len + int(local_pos[t].item())
            mapping[t] = cache_row_from_table(table, logical_pos)
        return mapping
    def init_ori_block_table():
        table = torch.full((SPARSE_ORI_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(SPARSE_ORI_MAX_BLOCKS):
            table[block] = block
        return table
    def init_cmp_kv():
        cache = torch.zeros(HCA_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_CMP_BLOCK_NUM * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()
        completed = context_len // COMPRESS_RATIO
        if completed > 0:
            prefix_cmp = ((torch.rand(completed, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for cmp_slot in range(completed):
                row = cache_row_from_table(table, cmp_slot, block_size=CMP_STORAGE_BLOCK_SIZE)
                if row >= 0:
                    cache_flat[row] = prefix_cmp[cmp_slot]
        return cache
    def init_cmp_block_table():
        # Single-request paged table: one compressed page mapped to physical block 0.
        table = torch.full((SPARSE_CMP_MAX_BLOCKS,), -1, dtype=torch.int32)
        table[0] = 0
        return table
    def init_position_ids():
        return token_meta()[1]
    def init_cmp_slot_mapping():
        out = torch.full((T,), -1, dtype=torch.int64)
        table = init_cmp_block_table()
        records = cmp_write_records()
        for token_id, cmp_slot in records:
            out[token_id] = cache_row_from_table(
                table, cmp_slot, block_size=CMP_STORAGE_BLOCK_SIZE
            )
        return out
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        _, pos = token_meta()
        for t in range(num_tokens):
            mapping[t] = state_row(int(pos[t].item()))
        return mapping
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5
    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        # Compressor recurrent state is written in-place but not validated here
        # (decode parity); cmp_kv is validated with the child precision contract.
        TensorSpec(
            "compress_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec("compress_state_block_table", [HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec(
            "kv_cache",
            [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
        ),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("ori_block_table", [SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec(
            "cmp_kv",
            [HCA_CMP_BLOCK_NUM, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", [SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]




# model config

# CP layout
LOCAL_PARTS = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
SEGMENT_ROWS = MAX_SEGMENT_TILES * TAIL_ROWS
ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_SIZE
RAW_BLOCKS_DYN = pl.dynamic("CP_HCA_RAW_BLOCKS_DYN")
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2
MAX_COMPRESSED_ROWS_PER_SEGMENT = (MAX_SEGMENT_TILES * TAIL_ROWS // COMPRESS_RATIO)
assert CMP_ROWS_PER_SEGMENT == MAX_COMPRESSED_ROWS_PER_SEGMENT
MAX_COMPRESS_LEAVES = 1 + MAX_SEGMENT_TILES
# Concurrent scalar stores must not share a 64-byte DDR cache line.
LEAF_NUM_TOKENS_STRIDE = 16
ROWS_PER_AUGMENTED_PART = MAX_COMPRESS_LEAVES * TAIL_ROWS
LOCAL_AUGMENTED_ROWS = LOCAL_PARTS * ROWS_PER_AUGMENTED_PART
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
# A compressor leaf emits at most one row in the persistent HCA page layout.
LEAF_CMP_BLOCKS = 1
LEAF_CMP_ROWS = (
    LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS * CMP_STORAGE_BLOCK_SIZE
)
STATE_ROWS = HCA_STATE_BLOCK_NUM * HCA_STATE_BLOCK_SIZE

# Canonical per-dispatch ring sizing for the standalone L3 harness.  The
# dbdd runtime no longer reads the retired PTO2_RING_* environment variables.
PREFILL_CP_HCA_RING_HEAP = (1024 * 1024 * 1024,) * 4

def active_tile(segment_len: int, tile: int) -> int:
    return max(0, min(TAIL_ROWS, segment_len - tile * TAIL_ROWS))


def segment_starts(prefix: int, span: int, nseg: int):
    return [prefix + segment * span for segment in range(nseg)]


def owner_segments(cp_size: int):
    owners = [[-1, -1] for _ in range(cp_size)]
    for segment in range(2 * cp_size):
        owners[cp_owner_rank(segment, cp_size)][cp_owner_part(segment, cp_size)] = segment
    return owners


def _tail_start(segment_start: int, segment_len: int) -> int:
    return segment_start + max(0, segment_len - TAIL_ROWS)


def _ring_phys_row(position: int) -> int:
    return position % ORI_CACHE_ROWS


def _lower_raw_key(
    key_abs: int,
    segment: int,
    tile: int,
    starts: list[int],
    lengths: list[int],
    prefix: int,
) -> int:
    segment_start = starts[segment]
    tile_start = segment_start + tile * TAIL_ROWS
    tile_len = active_tile(lengths[segment], tile)
    if key_abs < prefix:
        return _ring_phys_row(key_abs)
    if tile_start <= key_abs < tile_start + tile_len:
        return OVERLAY_BASE + TAIL_ROWS + key_abs - tile_start
    if key_abs >= tile_start:
        return -1
    if tile == 0:
        predecessor = segment - 1
        if predecessor < 0:
            return -1
        predecessor_start = _tail_start(starts[predecessor], lengths[predecessor])
        predecessor_len = min(TAIL_ROWS, lengths[predecessor])
    else:
        predecessor_start = tile_start - TAIL_ROWS
        predecessor_len = active_tile(lengths[segment], tile - 1)
    if predecessor_start <= key_abs < predecessor_start + predecessor_len:
        return OVERLAY_BASE + key_abs - predecessor_start
    return -1


def _build_raw_attention_metadata(cp_size: int, *, num_tokens: int | None = None):
    import torch

    prefix = 0
    if num_tokens is None:
        num_tokens = 2 * cp_size * MAX_SEGMENT_TILES * TAIL_ROWS
    span, starts, lengths = cp_segment_layout(num_tokens, cp_size)
    owners = owner_segments(cp_size)
    query_positions = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, dtype=torch.int32
    )
    query_requests = torch.full_like(query_positions, -1)
    overlay_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS),
        -1,
        dtype=torch.int32,
    )
    overlay_requests = torch.full_like(overlay_positions, -1)
    overlay_lengths = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        OVERLAY_SOURCES,
        dtype=torch.int32,
    )
    swa_indices = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN),
        -1,
        dtype=torch.int32,
    )
    segment_active = torch.zeros(cp_size, LOCAL_PARTS, dtype=torch.int32)
    predecessors = torch.full_like(segment_active, -1)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            segment_len = lengths[segment]
            segment_active[rank, part] = segment_len
            predecessors[rank, part] = segment - 1
            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(segment_len, tile)
                tile_start = starts[segment] + tile * TAIL_ROWS
                if active:
                    query_positions[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    query_requests[rank, part, tile, :active] = 0
                if tile == 0:
                    predecessor = segment - 1
                    predecessor_len = (
                        min(TAIL_ROWS, lengths[predecessor])
                        if predecessor >= 0
                        else 0
                    )
                    predecessor_start = (
                        _tail_start(starts[predecessor], lengths[predecessor])
                        if predecessor >= 0
                        else 0
                    )
                else:
                    predecessor_len = active_tile(segment_len, tile - 1)
                    predecessor_start = tile_start - TAIL_ROWS
                if predecessor_len:
                    overlay_positions[
                        rank, part, tile, :predecessor_len
                    ] = torch.arange(
                        predecessor_start,
                        predecessor_start + predecessor_len,
                        dtype=torch.int32,
                    )
                    overlay_requests[rank, part, tile, :predecessor_len] = 0
                if active:
                    overlay_positions[
                        rank, part, tile, TAIL_ROWS:TAIL_ROWS + active
                    ] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    overlay_requests[
                        rank, part, tile, TAIL_ROWS:TAIL_ROWS + active
                    ] = 0
                overlay_lengths[rank, part, tile, 0] = predecessor_len
                overlay_lengths[rank, part, tile, 1] = active
                for query_row in range(active):
                    query_abs = tile_start + query_row
                    for sparse_col in range(WIN):
                        key_abs = query_abs - WIN + 1 + sparse_col
                        if 0 <= key_abs <= query_abs:
                            swa_indices[
                                rank, part, tile, query_row, sparse_col
                            ] = _lower_raw_key(
                                key_abs,
                                segment,
                                tile,
                                starts,
                                lengths,
                                prefix,
                            )
    final_seg_src, final_row_src = cp_final_window_sources(lengths)
    final_slot_mapping = torch.tensor(
        [
            _ring_phys_row(prefix + sum(lengths) - TAIL_ROWS + row)
            if sum(lengths) - TAIL_ROWS + row >= 0 else -1
            for row in range(TAIL_ROWS)
        ],
        dtype=torch.int32,
    )
    return {
        "segment_active_lengths": segment_active,
        "predecessor_segments": predecessors,
        "query_position_ids": query_positions,
        "query_token_to_request": query_requests,
        "overlay_position_ids": overlay_positions,
        "overlay_token_to_request": overlay_requests,
        "overlay_active_lengths": overlay_lengths,
        "swa_indices": swa_indices,
        "reverse_index": cp_reverse_index(cp_size).to(torch.int32),
        "final_win_seg_src": final_seg_src.to(torch.int32),
        "final_win_row_src": final_row_src.to(torch.int32),
        "final_slot_mapping": final_slot_mapping,
    }


def _cmp_slot(boundary_position: int) -> int:
    if (boundary_position + 1) % COMPRESS_RATIO:
        return -1
    return (boundary_position + 1) // COMPRESS_RATIO - 1


def _cmp_block_tables(cp_size: int):
    import torch

    tables = torch.full(
        (cp_size, PREFILL_CMP_MAX_BLOCKS), -1, dtype=torch.int32
    )
    for rank in range(cp_size):
        for logical_block in range(PREFILL_CMP_MAX_BLOCKS):
            physical = logical_block % CP_PREFILL_CMP_BLOCK_NUM
            tables[rank, logical_block] = physical
    return tables


def _state_block_tables(cp_size: int):
    import torch

    tables = torch.empty(
        cp_size, HCA_STATE_MAX_BLOCKS, dtype=torch.int32
    )
    for rank in range(cp_size):
        for logical_block in range(HCA_STATE_MAX_BLOCKS):
            tables[rank, logical_block] = (
                logical_block * 17 + 3
            ) % HCA_STATE_PHYSICAL_BLOCKS
    return tables


def build_hca_metadata(cp_size: int = CP_SIZE, *, num_tokens: int | None = None):
    """Build canonical zero-history CP-HCA metadata."""
    import torch

    if cp_size not in CP_CHOICES:
        raise ValueError(f"cp_size must be one of {CP_CHOICES}, got {cp_size}")

    prefix = 0
    if num_tokens is None:
        num_tokens = 2 * cp_size * MAX_SEGMENT_TILES * TAIL_ROWS
    span, starts, lengths = cp_segment_layout(num_tokens, cp_size)
    owners = owner_segments(cp_size)

    query_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS),
        -1,
        dtype=torch.int32,
    )
    query_active = torch.zeros(
        cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, dtype=torch.int32
    )
    cmp_indices = torch.full(
        (
            cp_size,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            IDX_TOPK,
        ),
        -1,
        dtype=torch.int32,
    )
    segment_cmp_positions = torch.full(
        (cp_size, LOCAL_PARTS, MAX_COMPRESSED_ROWS_PER_SEGMENT),
        -1,
        dtype=torch.int32,
    )
    segment_cmp_slots = torch.full_like(segment_cmp_positions, -1)
    snapshot_positions = torch.full(
        (cp_size, LOCAL_PARTS, TAIL_ROWS), -1, dtype=torch.int32
    )
    snapshot_valid = torch.zeros(
        cp_size, LOCAL_PARTS, dtype=torch.int32
    )
    segment_tail_positions = torch.full(
        (2 * cp_size, TAIL_ROWS), -1, dtype=torch.int32
    )

    for segment in range(2 * cp_size):
        segment_end = starts[segment] + lengths[segment]
        valid = min(TAIL_ROWS, lengths[segment])
        if valid:
            tail_begin = segment_end - valid
            segment_tail_positions[segment, :valid] = torch.arange(
                tail_begin, segment_end, dtype=torch.int32
            )

    for rank, rank_segments in enumerate(owners):
        for part, segment in enumerate(rank_segments):
            segment_start = starts[segment]
            segment_len = lengths[segment]
            segment_end = segment_start + segment_len

            boundaries = [
                position
                for position in range(segment_start, segment_end)
                if _cmp_slot(position) >= 0
            ]
            if len(boundaries) > MAX_COMPRESSED_ROWS_PER_SEGMENT:
                raise ValueError(
                    f"segment {segment} has {len(boundaries)} compressed rows; "
                    f"capacity is {MAX_COMPRESSED_ROWS_PER_SEGMENT}"
                )
            for index, boundary in enumerate(boundaries):
                segment_cmp_positions[rank, part, index] = boundary
                segment_cmp_slots[rank, part, index] = _cmp_slot(boundary)

            live_valid = min(TAIL_ROWS, segment_end) if segment_len else 0
            live_start = segment_end - live_valid
            snapshot_valid[rank, part] = live_valid
            if live_valid:
                snapshot_positions[rank, part, :live_valid] = torch.arange(
                    live_start, segment_end, dtype=torch.int32
                )

            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(segment_len, tile)
                query_active[rank, part, tile] = active
                tile_start = segment_start + tile * TAIL_ROWS
                if active:
                    query_positions[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                for row in range(active):
                    absolute_position = tile_start + row
                    visible = min(
                        IDX_TOPK,
                        (absolute_position + 1) // COMPRESS_RATIO,
                    )
                    if visible:
                        cmp_indices[rank, part, tile, row, :visible] = (
                            torch.arange(visible, dtype=torch.int32)
                        )

    active_segments = [
        segment for segment, length in enumerate(lengths) if length > 0
    ]
    if not active_segments:
        raise ValueError("CP-HCA requires at least one active logical segment")
    final_segment = active_segments[-1]
    final_owner_rank = next(
        rank
        for rank, rank_segments in enumerate(owners)
        if final_segment in rank_segments
    )
    final_owner_part = owners[final_owner_rank].index(final_segment)
    owner_rank_table, owner_part_table = cp_owner_tables(cp_size)

    metadata = {
        "cp_size": cp_size,
        "prefix": prefix,
        "segment_span": span,
        "segment_lengths": torch.tensor(lengths, dtype=torch.int32),
        "segment_starts": torch.tensor(starts, dtype=torch.int32),
        "owner_segments": torch.tensor(owners, dtype=torch.int32),
        "query_positions": query_positions,
        "query_active": query_active,
        "cmp_indices": cmp_indices,
        "segment_cmp_positions": segment_cmp_positions,
        "segment_cmp_slots": segment_cmp_slots,
        "snapshot_positions": snapshot_positions,
        "snapshot_valid": snapshot_valid,
        "segment_tail_positions": segment_tail_positions,
        "final_segment": final_segment,
        "final_segment_t": torch.tensor([final_segment], dtype=torch.int32),
        "final_owner_rank": final_owner_rank,
        "final_owner_part": final_owner_part,
        "owner_rank_table": owner_rank_table,
        "owner_part_table": owner_part_table,
        "cmp_block_table": _cmp_block_tables(cp_size),
        "compress_state_block_table": _state_block_tables(cp_size),
    }
    return metadata


@pl.jit.inline
def prefill_cp_hca_core(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
    ],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_BLOCKS_DYN,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [CP_CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    cache_owner_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tail_comm_epoch: pl.Scalar[pl.INT32],
    compact_comm_epoch_base: pl.Scalar[pl.INT32],
):
    """CP-HCA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit).

    ``tail_comm_epoch`` and ``compact_comm_epoch_base`` drive the shared
    cross-layer ready/consumed counters of the dual-tail and HCA compact
    domains respectively; local payload rows stay at 0 (``EPOCHS == 1``).
    Standalone/single-layer callers pass 0 for both, preserving behavior.
    """
    state_blocks = pl.tensor.dim(compress_state, 0)
    state_rows = state_blocks * HCA_STATE_BLOCK_SIZE
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_cos_il = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    local_kv = pl.create_tensor([LOCAL_ROWS, HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    qr = pl.create_tensor([LOCAL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([LOCAL_ROWS, 1], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    query_positions_flat = pl.reshape(query_positions, [LOCAL_ROWS])

    # Recipes treats the two owned 512-row segments as one rank-local 1024-row
    # query projection.  KV is projected later from the augmented hidden
    # sequence (predecessor128 + current512), after hidden-only CP exchange.
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        position_tile = pl.slice(query_positions_flat, [TAIL_ROWS], [row0])
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [row0, 0])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        materialize_rope_rows(
            freqs_cos, freqs_sin, position_tile, active,
            cos_tile, sin_tile,
        )
    prefill_attention_prolog(
        x_flat, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos_flat, rope_sin_flat,
        normed, post, comb, rope_cos_il, rope_sin_signed, rope_swap_idx, q, qr, qr_scale,
    )

    local_hidden_tail = pl.create_tensor([EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_tail_assemble"):
        for part in pl.range(LOCAL_PARTS):
            total = pl.read(segment_active_lengths, [part])
            tail_offset0 = pl.max(total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_offset0 + row
                destination = part * TAIL_ROWS + row
                local_hidden_tail[
                    destination : destination + 1, :
                ] = pl.full([1, D], dtype=pl.BF16, value=0.0)
                if tail_offset < total:
                    source = (part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset)
                    local_hidden_tail[destination : destination + 1, :] = normed[source : source + 1, :]

    logical_hidden = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_hidden_tail_exchange") as tail_exchange_tid:
        _prefill_cp_hidden_tail_exchange_wave(
            local_hidden_tail,
            reverse_index, owner_rank_table,
            hidden_tail_window, tail_ready, tail_consumed,
            logical_hidden,
            my_rank, pl.cast(0, pl.INT32), tail_comm_epoch,
        )

    effective_x = pl.create_tensor([LOCAL_AUGMENTED_ROWS, D], dtype=pl.BF16)
    leaf_positions = pl.create_tensor(
        [LOCAL_AUGMENTED_ROWS],
        dtype=pl.INT32,
    )
    leaf_num_tokens = pl.create_tensor(
        [LOCAL_PARTS, LEAF_NUM_TOKENS_STRIDE],
        dtype=pl.INT32,
    )
    leaf_cmp_slots = pl.create_tensor(
        [LOCAL_AUGMENTED_ROWS],
        dtype=pl.INT64,
    )
    leaf_state_slots = pl.create_tensor(
        [LOCAL_AUGMENTED_ROWS],
        dtype=pl.INT64,
    )
    with pl.spmd(
        LOCAL_PARTS,
        name_hint="cp_hca_leaf_lowering",
        deps=[tail_exchange_tid],
    ) as leaf_lowering_tid:
        part = pl.tile.get_block_idx()
        predecessor = pl.read(predecessor_segments, [part])
        predecessor_valid = pl.read(overlay_active_lengths, [part, 0, 0])
        leaf_token_row = pl.full([1, LEAF_NUM_TOKENS_STRIDE], dtype=pl.INT32, value=0)
        pl.write(leaf_token_row, [0, 0], predecessor_valid)
        for leaf_row in pl.range(TAIL_ROWS):
            leaf_index = part * MAX_COMPRESS_LEAVES * TAIL_ROWS + leaf_row
            effective_x[leaf_index:leaf_index + 1, :] = pl.full(
                [1, D], dtype=pl.BF16, value=0.0
            )
            pl.write(leaf_positions, [leaf_index], pl.cast(0, pl.INT32))
            pl.write(leaf_cmp_slots, [leaf_index], pl.cast(-1, pl.INT64))
            pl.write(leaf_state_slots, [leaf_index], pl.cast(-1, pl.INT64))
            if predecessor >= 0 and leaf_row < predecessor_valid:
                source = predecessor * TAIL_ROWS + leaf_row
                effective_x[leaf_index:leaf_index + 1, :] = logical_hidden[source:source + 1, :]
                position = pl.read(segment_tail_positions, [predecessor, leaf_row])
                pl.write(leaf_positions, [leaf_index], position)
                if position >= 0:
                    logical_block = position // HCA_STATE_BLOCK_SIZE
                    physical_block = pl.read(compress_state_block_table, [logical_block])
                    if physical_block >= 0:
                        predecessor_state_row = (
                            pl.cast(physical_block, pl.INT64)
                            * HCA_STATE_BLOCK_SIZE
                            + position % HCA_STATE_BLOCK_SIZE
                        )
                        pl.write(leaf_state_slots, [leaf_index], predecessor_state_row)
        for tile in pl.range(MAX_SEGMENT_TILES):
            leaf = 1 + tile
            active = pl.read(overlay_active_lengths, [part, tile, 1])
            pl.write(leaf_token_row, [0, leaf], active)
            local_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
            leaf_row0 = (part * MAX_COMPRESS_LEAVES + leaf) * TAIL_ROWS
            for leaf_row in pl.range(TAIL_ROWS):
                destination = leaf_row0 + leaf_row
                effective_x[destination:destination + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
                pl.write(leaf_positions, [destination], pl.cast(0, pl.INT32))
                pl.write(leaf_cmp_slots, [destination], pl.cast(-1, pl.INT64))
                pl.write(leaf_state_slots, [destination], pl.cast(-1, pl.INT64))
                if leaf_row < active:
                    source = local_row0 + leaf_row
                    effective_x[destination:destination + 1, :] = normed[source:source + 1, :]
                    position = pl.read(query_positions_flat, [source])
                    pl.write(leaf_positions, [destination], position)
                    logical_block = position // HCA_STATE_BLOCK_SIZE
                    physical_block = pl.read(
                        compress_state_block_table, [logical_block]
                    )
                    if physical_block >= 0:
                        local_state_row = (
                            pl.cast(physical_block, pl.INT64)
                            * HCA_STATE_BLOCK_SIZE
                            + position % HCA_STATE_BLOCK_SIZE
                        )
                        pl.write(
                            leaf_state_slots,
                            [destination],
                            local_state_row,
                        )
                    if (position + 1) % COMPRESS_RATIO == 0:
                        pl.write(leaf_cmp_slots, [destination], pl.cast(0, pl.INT64))
        leaf_num_tokens[part:part + 1, 0:LEAF_NUM_TOKENS_STRIDE] = leaf_token_row

    # Recipes projects KV locally from the augmented hidden sequence rather
    # than exchanging projected KV.  `effective_x` is already laid out as
    # [predecessor128, current512] for each owned segment and is shared with
    # the HCA compressor, so no second hidden gather is needed.
    augmented_rope_cos = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    augmented_rope_sin = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    augmented_rope_cos_il = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    augmented_rope_sin_signed = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    augmented_rope_swap_idx = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    augmented_kv = pl.create_tensor([LOCAL_AUGMENTED_ROWS, HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        leaf_positions,
        pl.const(LOCAL_AUGMENTED_ROWS, pl.INT32),
        augmented_rope_cos,
        augmented_rope_sin,
    )
    rope_prepare(
        augmented_rope_cos,
        augmented_rope_sin,
        augmented_rope_cos_il,
        augmented_rope_sin_signed,
        augmented_rope_swap_idx,
    )
    kv_proj_rope(
        effective_x,
        wkv,
        gamma_ckv,
        augmented_rope_cos_il,
        augmented_rope_sin_signed,
        augmented_rope_swap_idx,
        augmented_kv,
        leaf_lowering_tid,
    )

    logical_kv = pl.create_tensor([EPOCHS * CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_augmented_kv_scatter"):
        for part in pl.range(LOCAL_PARTS):
            augmented_row0 = part * ROWS_PER_AUGMENTED_PART
            local_row0 = part * MAX_SEGMENT_TILES * TAIL_ROWS
            for row0 in pl.range(0, MAX_SEGMENT_TILES * TAIL_ROWS, ROW_TILE):
                local_kv[
                    local_row0 + row0:local_row0 + row0 + ROW_TILE,
                    :,
                ] = augmented_kv[
                    augmented_row0 + TAIL_ROWS + row0:
                    augmented_row0 + TAIL_ROWS + row0 + ROW_TILE,
                    :,
                ]
            predecessor = pl.read(predecessor_segments, [part])
            if predecessor >= 0:
                predecessor_row0 = predecessor * TAIL_ROWS
                for row0 in pl.range(0, TAIL_ROWS, ROW_TILE):
                    logical_kv[
                        predecessor_row0 + row0:
                        predecessor_row0 + row0 + ROW_TILE,
                        :,
                    ] = augmented_kv[
                        augmented_row0 + row0:
                        augmented_row0 + row0 + ROW_TILE,
                        :,
                    ]

    # The decode window is also projected locally from gathered hidden tails
    # in Recipes.  It may span the final two logical segments, so construct it
    # row-by-row from the canonical source metadata before one 128-row KV pass.
    final_hidden = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
    final_positions = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_hca_final_hidden_lowering",
        deps=[tail_exchange_tid],
    ) as final_hidden_tid:
        for row in pl.range(TAIL_ROWS):
            final_hidden[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            pl.write(final_positions, [row], pl.cast(0, pl.INT32))
            segment = pl.read(final_win_seg_src, [row])
            source_row = pl.read(final_win_row_src, [row])
            if segment >= 0 and source_row >= 0:
                source = segment * TAIL_ROWS + source_row
                final_hidden[row:row + 1, :] = logical_hidden[source:source + 1, :]
                position = pl.read(segment_tail_positions, [segment, source_row])
                if position >= 0:
                    pl.write(final_positions, [row], position)

    final_rope_cos = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    final_rope_sin = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    final_rope_cos_il = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    final_rope_sin_signed = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    final_rope_swap_idx = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    final_kv = pl.create_tensor([TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        final_positions,
        pl.const(TAIL_ROWS, pl.INT32),
        final_rope_cos,
        final_rope_sin,
    )
    rope_prepare(
        final_rope_cos,
        final_rope_sin,
        final_rope_cos_il,
        final_rope_sin_signed,
        final_rope_swap_idx,
    )
    kv_proj_rope(
        final_hidden,
        wkv,
        gamma_ckv,
        final_rope_cos_il,
        final_rope_sin_signed,
        final_rope_swap_idx,
        final_kv,
        final_hidden_tid,
    )

    scratch_state = pl.create_tensor(
        [
            LOCAL_PARTS * state_blocks,
            HCA_STATE_BLOCK_SIZE,
            COMPRESS_STATE_DIM,
        ],
        dtype=pl.FP32,
    )
    persistent_state_flat = pl.reshape(compress_state, [state_rows, COMPRESS_STATE_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * state_rows, COMPRESS_STATE_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_seed_state"):
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            for state_row in pl.range(state_rows):
                destination = part * state_rows + state_row
                scratch_state_flat[
                    destination : destination + 1, :
                ] = pl.full(
                    [1, COMPRESS_STATE_DIM], dtype=pl.FP32, value=0.0
                )
                if segment == 0 and pl.read(segment_starts_t, [0]) > 0:
                    persistent_state_row = persistent_state_flat[state_row : state_row + 1, :]
                    scratch_state_flat[destination : destination + 1, :] = persistent_state_row

    leaf_cmp = pl.create_tensor(
        [
            LOCAL_PARTS * MAX_COMPRESS_LEAVES * LEAF_CMP_BLOCKS,
            CMP_STORAGE_BLOCK_SIZE,
            1,
            HEAD_DIM,
        ],
        dtype=pl.BF16,
    )
    for part in pl.range(LOCAL_PARTS):
        state_base = part * state_blocks
        state_part = pl.slice(
            scratch_state,
            [
                state_blocks,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            [state_base, 0, 0],
        )
        for leaf in pl.range(MAX_COMPRESS_LEAVES):
            leaf_index = part * MAX_COMPRESS_LEAVES + leaf
            token0 = leaf_index * TAIL_ROWS
            cmp_block0 = leaf_index * LEAF_CMP_BLOCKS
            x_leaf = pl.slice(effective_x, [TAIL_ROWS, D], [token0, 0])
            position_leaf = pl.slice(leaf_positions, [TAIL_ROWS], [token0])
            cmp_slots_leaf = pl.slice(leaf_cmp_slots, [TAIL_ROWS], [token0])
            state_slots_leaf = pl.slice(leaf_state_slots, [TAIL_ROWS], [token0])
            cmp_leaf = pl.slice(
                leaf_cmp,
                [LEAF_CMP_BLOCKS, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM],
                [cmp_block0, 0, 0, 0],
            )
            active = pl.read(leaf_num_tokens, [part, leaf])
            cmp_leaf, state_part = prefill_compressor_ratio128(
                x_leaf, state_part, compress_state_block_table,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                freqs_cos, freqs_sin,
                cmp_leaf, position_leaf, active,
                cmp_slots_leaf, state_slots_leaf,
            )
            leaf_cmp = pl.assemble(leaf_cmp, cmp_leaf, [cmp_block0, 0, 0, 0])
        # Publish the updated view in fixed-size rows; the physical pool
        # extent is runtime-sized and cannot form one on-chip tile.
        state_part_flat = pl.reshape(state_part, [state_rows, COMPRESS_STATE_DIM])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_state_part_publish"):
            for row in pl.range(state_rows):
                destination = part * state_rows + row
                scratch_state_flat[destination:destination + 1, :] = state_part_flat[row:row + 1, :]

    local_cmp_payload = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], dtype=pl.BF16)
    local_cmp_meta = pl.create_tensor([EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], dtype=pl.INT32)
    local_state_payload = pl.create_tensor([EPOCHS * TAIL_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32)
    local_state_meta = pl.create_tensor([EPOCHS, STATE_META_DIM], dtype=pl.INT32)
    leaf_cmp_flat = pl.reshape(leaf_cmp, [LEAF_CMP_ROWS, HEAD_DIM])
    scratch_state_flat = pl.reshape(scratch_state, [LOCAL_PARTS * state_rows, COMPRESS_STATE_DIM])
    final_segment = pl.read(final_segment_t, [0])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_pack_compact") as pack_compact_tid:
        for record in pl.range(EPOCHS * CMP_ROWS_PER_RANK):
            local_cmp_payload[record : record + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
            for field in pl.range(CMP_META_DIM):
                pl.write(local_cmp_meta, [record, field], pl.cast(-1, pl.INT32))
        for record in pl.range(EPOCHS * TAIL_ROWS):
            local_state_payload[record : record + 1, :] = pl.full(
                [1, COMPRESS_STATE_DIM], dtype=pl.FP32, value=0.0
            )
        for record in pl.range(EPOCHS):
            for field in pl.range(STATE_META_DIM):
                pl.write(local_state_meta, [record, field], pl.cast(-1, pl.INT32))
        for part in pl.range(LOCAL_PARTS):
            segment = pl.read(owner_segments_t, [part])
            for tile in pl.range(MAX_SEGMENT_TILES):
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                query_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
                for row in pl.range(TAIL_ROWS):
                    if row < active:
                        position = pl.read(
                            query_positions_flat, [query_row0 + row]
                        )
                        if (position + 1) % COMPRESS_RATIO == 0:
                            destination = part * CMP_ROWS_PER_SEGMENT + tile
                            leaf_index = (
                                part * MAX_COMPRESS_LEAVES + 1 + tile
                            )
                            source = (
                                leaf_index
                                * LEAF_CMP_BLOCKS
                                * CMP_STORAGE_BLOCK_SIZE
                            )
                            local_cmp_payload[
                                destination : destination + 1, :
                            ] = leaf_cmp_flat[source : source + 1, :]
                            pl.write(
                                local_cmp_meta,
                                [destination, 0],
                                pl.cast(1, pl.INT32),
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 1],
                                segment,
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 2],
                                position,
                            )
                            pl.write(
                                local_cmp_meta,
                                [destination, 3],
                                pl.cast(
                                    (position + 1) // COMPRESS_RATIO - 1,
                                    pl.INT32,
                                ),
                            )
            if segment == final_segment:
                valid = pl.read(snapshot_valid, [part])
                end_position = (
                    pl.read(segment_starts_t, [segment])
                    + pl.read(segment_active_lengths, [part])
                )
                pl.write(
                    local_state_meta, [0, 0], pl.cast(1, pl.INT32)
                )
                pl.write(local_state_meta, [0, 1], segment)
                pl.write(local_state_meta, [0, 2], valid)
                pl.write(local_state_meta, [0, 3], end_position)
                for row in pl.range(TAIL_ROWS):
                    if row < valid:
                        position = pl.read(snapshot_positions, [part, row])
                        logical_block = position // HCA_STATE_BLOCK_SIZE
                        physical_block = pl.read(
                            compress_state_block_table, [logical_block]
                        )
                        if physical_block >= 0:
                            source = (
                                part * state_rows
                                + pl.cast(physical_block, pl.INDEX)
                                * HCA_STATE_BLOCK_SIZE
                                + position % HCA_STATE_BLOCK_SIZE
                            )
                            local_state_payload[
                                row:row + 1, :
                            ] = scratch_state_flat[source:source + 1, :]

    attn_cmp_flat = pl.create_tensor([HCA_MAX_COMPRESSED_ROWS, HEAD_DIM], dtype=pl.BF16)
    attn_cmp_table = pl.create_tensor([PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_attn_cache_init"):
        attn_cmp_flat[:, :] = pl.full([HCA_MAX_COMPRESSED_ROWS, HEAD_DIM], dtype=pl.BF16, value=0.0)
        for logical in pl.range(PREFILL_CMP_MAX_BLOCKS):
            pl.write(attn_cmp_table, [logical], pl.cast(logical, pl.INT32))
    attn_cmp_kv = pl.reshape(attn_cmp_flat, [HCA_MAX_COMPRESSED_ROWS, 1, 1, HEAD_DIM])
    cmp_cache_rows = pl.tensor.dim(cmp_kv, 0)
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    compact_commit_tid = _prefill_cp_hca_compact_exchange_commit_wave(
        local_cmp_payload,
        local_cmp_meta,
        local_state_payload,
        local_state_meta,
        owner_rank_table,
        owner_part_table,
        cmp_block_table,
        compress_state_block_table,
        cmp_window,
        cmp_meta_window,
        state_window,
        state_meta_window,
        compact_ready,
        compact_consumed,
        cmp_kv_flat,
        compress_state,
        attn_cmp_flat, cache_owner_rank,
        my_rank,
        pl.cast(0, pl.INT32),
        compact_comm_epoch_base,
        pack_compact_tid,
    )

    raw_blocks = pl.tensor.dim(kv_cache, 0)
    raw_rows = raw_blocks * BLOCK_SIZE
    cache_flat = pl.reshape(kv_cache, [raw_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_hca_raw_commit") as raw_commit_tid:
        for row in pl.range(TAIL_ROWS):
            raw_segment = pl.read(final_win_seg_src, [row])
            raw_source_row = pl.read(final_win_row_src, [row])
            raw_destination = pl.read(final_slot_mapping, [row])
            if my_rank == cache_owner_rank and raw_segment >= 0 and raw_source_row >= 0 and raw_destination >= 0 and raw_destination < raw_rows:
                cache_flat[raw_destination:raw_destination + 1, :] = final_kv[row:row + 1, :]

    # A TaskId-only fence is insufficient on PTOAS 0.60 if it does not carry
    # real producer reads.  Sample Q and augmented KV while joining the compact
    # cache commit.  Native attention lowers the Recipes physical raw-cache
    # layout analytically, without materializing an intermediate index table.
    native_ready_anchor = pl.create_tensor([2], dtype=pl.BF16)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_hca_native_cache_ready",
        deps=[compact_commit_tid],
    ) as native_ready_tid:
        pl.write(native_ready_anchor, [0], pl.read(q, [0, 0, 0]))
        pl.write(native_ready_anchor, [1], pl.read(augmented_kv, [0, 0]))

    part0_active = pl.read(segment_active_lengths, [0])
    part1_active = pl.read(segment_active_lengths, [1])
    part0_rows = pl.max(1, pl.min(SEGMENT_ROWS, part0_active))
    part1_rows = pl.max(1, pl.min(SEGMENT_ROWS, part1_active))
    attn_out = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
    q_part0 = pl.slice(q, [part0_rows, H, HEAD_DIM], [0, 0, 0])
    full_kv_part0_flat = pl.slice(augmented_kv, [ROWS_PER_AUGMENTED_PART, HEAD_DIM], [0, 0])
    full_kv_part0 = pl.reshape(full_kv_part0_flat, [MAX_COMPRESS_LEAVES, BLOCK_SIZE, 1, HEAD_DIM])
    positions_part0 = pl.slice(query_positions_flat, [part0_rows], [0])
    cos_part0 = pl.slice(rope_cos_flat, [part0_rows, ROPE_DIM], [0, 0])
    sin_part0 = pl.slice(rope_sin_flat, [part0_rows, ROPE_DIM], [0, 0])
    attn_out_part0 = pl.slice(attn_out, [part0_rows, D], [0, 0])
    part0_predecessor_valid = pl.read(overlay_active_lengths, [0, 0, 0])
    if pl.read(predecessor_segments, [0]) < 0:
        part0_predecessor_valid = pl.cast(0, pl.INT32)
    part0_attn_tid = hca_attn(
        q_part0,
        full_kv_part0,
        part0_predecessor_valid,
        attn_cmp_kv,
        attn_cmp_table,
        positions_part0,
        attn_sink,
        cos_part0,
        sin_part0,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out_part0,
        part0_active,
        native_ready_tid,
        native_ready_tid,
    )

    part1_row0 = SEGMENT_ROWS
    part1_augmented_row0 = ROWS_PER_AUGMENTED_PART
    q_part1 = pl.slice(q, [part1_rows, H, HEAD_DIM], [part1_row0, 0, 0])
    full_kv_part1_flat = pl.slice(
        augmented_kv,
        [ROWS_PER_AUGMENTED_PART, HEAD_DIM],
        [part1_augmented_row0, 0],
    )
    full_kv_part1 = pl.reshape(full_kv_part1_flat, [MAX_COMPRESS_LEAVES, BLOCK_SIZE, 1, HEAD_DIM])
    positions_part1 = pl.slice(query_positions_flat, [part1_rows], [part1_row0])
    cos_part1 = pl.slice(rope_cos_flat, [part1_rows, ROPE_DIM], [part1_row0, 0])
    sin_part1 = pl.slice(rope_sin_flat, [part1_rows, ROPE_DIM], [part1_row0, 0])
    attn_out_part1 = pl.slice(attn_out, [part1_rows, D], [part1_row0, 0])
    part1_predecessor_valid = pl.read(overlay_active_lengths, [1, 0, 0])
    if pl.read(predecessor_segments, [1]) < 0:
        part1_predecessor_valid = pl.cast(0, pl.INT32)
    attention_done_tid = hca_attn(
        q_part1,
        full_kv_part1,
        part1_predecessor_valid,
        attn_cmp_kv,
        attn_cmp_table,
        positions_part1,
        attn_sink,
        cos_part1,
        sin_part1,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out_part1,
        part1_active,
        native_ready_tid,
        native_ready_tid,
    )

    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        row0 = tile * TAIL_ROWS
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [row0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [row0, 0])
        residual_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [row0, 0, 0])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        attn_out_tile = pl.slice(
            attn_out, [TAIL_ROWS, D], [row0, 0]
        )
        y_tile = pl.create_tensor([TAIL_ROWS, HC_MULT, D], dtype=pl.FP32)
        hc_post_prefill(
            attn_out_tile, residual_tile,
            post_tile, comb_tile,
            y_tile, active,
        )
        x_out_flat[row0 : row0 + TAIL_ROWS, 0:HC_MULT, 0:D] = y_tile

    completion_token = pl.create_tensor([NUM_LOCAL_TILES, 1, 8], dtype=pl.FP32)
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_hca_rank_complete",
        deps=[tail_exchange_tid, compact_commit_tid, raw_commit_tid,
              part0_attn_tid, attention_done_tid],
        allow_early_resolve=False,
    ):
        for tile in pl.range(NUM_LOCAL_TILES):
            completion_token[tile : tile + 1, 0:1, 0:8] = pl.slice(
                x_out_flat, [1, 1, 8], [tile * TAIL_ROWS, 0, 0]
            )
    _completed = pl.read(completion_token, [0, 0, 0])

    return pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])


@pl.jit
def prefill_cp_hca_rank(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
    ],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_BLOCKS_DYN,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [CP_CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Standalone CP-HCA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    return prefill_cp_hca_core(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        segment_starts_t, segment_active_lengths,
        owner_segments_t, predecessor_segments,
        query_positions, query_requests,
        overlay_positions, overlay_requests,
        overlay_active_lengths, swa_indices, cmp_indices,
        segment_tail_positions,
        snapshot_positions, snapshot_valid, final_segment_t,
        reverse_index, owner_rank_table, owner_part_table,
        final_win_seg_src, final_win_row_src, final_slot_mapping,
        hidden_tail_window, kv_tail_window,
        tail_ready, tail_consumed,
        cmp_window, cmp_meta_window,
        state_window, state_meta_window,
        compact_ready, compact_consumed,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, pl.read(cache_owner_rank_t, [0]), my_rank,
        pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
    )


@pl.jit.host
def prefill_cp_hca_test(
    x_hc: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            HC_MULT,
            D,
        ],
        pl.FP32,
    ],
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                HCA_STATE_BLOCKS_DYN,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [CP_SIZE, HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                CP_CMP_BLOCK_NUM_DYN,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[
        [CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32
    ],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    query_positions: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
        ],
        pl.INT32,
    ],
    query_requests: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
        ],
        pl.INT32,
    ],
    overlay_positions: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_ROWS,
        ],
        pl.INT32,
    ],
    overlay_requests: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_ROWS,
        ],
        pl.INT32,
    ],
    overlay_active_lengths: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            OVERLAY_SOURCES,
        ],
        pl.INT32,
    ],
    swa_indices: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            WIN,
        ],
        pl.INT32,
    ],
    cmp_indices: pl.Tensor[
        [
            CP_SIZE,
            LOCAL_PARTS,
            MAX_SEGMENT_TILES,
            TAIL_ROWS,
            IDX_TOPK,
        ],
        pl.INT32,
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_valid: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[
        pl.Tensor[
            [
                CP_SIZE,
                LOCAL_PARTS,
                MAX_SEGMENT_TILES,
                TAIL_ROWS,
                HC_MULT,
                D,
            ],
            pl.FP32,
        ]
    ],
):
    hidden_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
    )
    kv_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    cmp_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    cmp_meta_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
    )
    state_window_buf = pld.alloc_window_buffer(
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32
    )
    state_meta_window_buf = pld.alloc_window_buffer(
        [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
    )
    compact_ready_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    compact_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(
            hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        kv_tail_window = pld.window(
            kv_tail_buf,
            [CP_TAIL_WINDOW_ROWS, HEAD_DIM],
            dtype=pl.BF16,
        )
        tail_ready = pld.window(
            tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        cmp_window = pld.window(
            cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        cmp_meta_window = pld.window(
            cmp_meta_window_buf,
            [CMP_WINDOW_ROWS, CMP_META_DIM],
            dtype=pl.INT32,
        )
        state_window = pld.window(
            state_window_buf,
            [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM],
            dtype=pl.FP32,
        )
        state_meta_window = pld.window(
            state_meta_window_buf,
            [CP_SIZE, STATE_META_DIM],
            dtype=pl.INT32,
        )
        compact_ready = pld.window(
            compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        compact_consumed = pld.window(
            compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        prefill_cp_hca_rank(
            x_hc[rank],
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
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            compress_state[rank],
            compress_state_block_table[rank],
            kv_cache[rank],
            cmp_kv[rank],
            cmp_block_table[rank],
            segment_starts_t,
            segment_active_lengths[rank],
            owner_segments_t[rank],
            predecessor_segments[rank],
            query_positions[rank],
            query_requests[rank],
            overlay_positions[rank],
            overlay_requests[rank],
            overlay_active_lengths[rank],
            swa_indices[rank],
            cmp_indices[rank],
            segment_tail_positions,
            snapshot_positions[rank],
            snapshot_valid[rank],
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
            compact_ready,
            compact_consumed,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            x_out[rank],
            cache_owner_rank_t[rank],
            rank,
            device=rank,
        )


def _state_physical_row(table, absolute_position: int) -> int:
    if absolute_position < 0 or absolute_position >= MAX_SEQ_LEN:
        return -1
    logical_block = absolute_position // HCA_STATE_BLOCK_SIZE
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return (
        physical_block * HCA_STATE_BLOCK_SIZE
        + absolute_position % HCA_STATE_BLOCK_SIZE
    )


def _cmp_physical_row(table, logical_slot: int) -> int:
    if logical_slot < 0:
        return -1
    logical_block = logical_slot // CMP_STORAGE_BLOCK_SIZE
    if logical_block >= table.numel():
        return -1
    physical_block = int(table[logical_block].item())
    if physical_block < 0:
        return -1
    return (
        physical_block * CMP_STORAGE_BLOCK_SIZE
        + logical_slot % CMP_STORAGE_BLOCK_SIZE
    )


def build_cp_tensor_specs(cp_size: int = CP_SIZE, *, num_tokens: int | None = None):
    """Build the canonical CP-HCA fixture."""
    import torch
    from golden import TensorSpec

    if cp_size != CP_SIZE:
        raise ValueError(
            f"runtime cp_size={cp_size} does not match static CP_SIZE={CP_SIZE}"
    )
    metadata = build_hca_metadata(cp_size, num_tokens=num_tokens)
    raw_metadata = _build_raw_attention_metadata(cp_size, num_tokens=num_tokens)
    torch.manual_seed(4100 + cp_size * 31)
    qkv_specs = {spec.name: spec for spec in build_qkv_tensor_specs(1, TAIL_ROWS)}
    sparse_specs = {
        spec.name: spec
        for spec in build_sparse_attn_tensor_specs(COMPRESS_RATIO, TAIL_ROWS)
    }
    compressor_specs = {
        spec.name: spec for spec in build_compressor_tensor_specs(0)
    }
    qkv_names = (
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
    )
    tail_names = ("attn_sink", "wo_a", "wo_b", "wo_b_scale")
    hca_values = {name: qkv_specs[name].create_tensor() for name in qkv_names}
    hca_values.update(
        {name: sparse_specs[name].create_tensor() for name in tail_names}
    )
    for source_name, target_name in (
        ("wkv", "cmp_wkv"),
        ("wgate", "cmp_wgate"),
        ("ape", "cmp_ape"),
        ("norm_w", "cmp_norm_w"),
    ):
        hca_values[target_name] = compressor_specs[source_name].create_tensor()
    hca_values["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    hca_values["hc_attn_scale"] = torch.randn(3)
    hca_values["hc_attn_base"] = torch.randn(MIX_HC)
    hca_values["attn_norm_w"] = torch.ones(D, dtype=torch.bfloat16)
    hca_values["freqs_cos"], hca_values["freqs_sin"] = (
        build_rope_tables(
            M, COMPRESS_RATIO, dtype=torch.bfloat16
        )
    )
    x_generator = torch.Generator().manual_seed(4100 + cp_size * 31)
    x_hc = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT,
        D,
        dtype=torch.float32,
    )
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    raw_metadata["overlay_active_lengths"][rank, part, tile, 1]
                )
                if active:
                    x_hc[rank, part, tile, :active].uniform_(
                        -1.0, 1.0, generator=x_generator
                    )
    kv_cache = torch.zeros(
        cp_size,
        ORI_MAX_BLOCKS,
        BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )

    common_names = (
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
        "cmp_wkv",
        "cmp_wgate",
        "cmp_ape",
        "cmp_norm_w",
    )
    specs = [
        TensorSpec(
            "x_hc",
            list(x_hc.shape),
            torch.float32,
            init_value=x_hc,
        )
    ]
    for name in common_names:
        value = hca_values[name]
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )

    state_tables = metadata["compress_state_block_table"]
    generator = torch.Generator().manual_seed(
        20260731 + cp_size * 101
    )
    state = torch.zeros(
        cp_size,
        HCA_STATE_PHYSICAL_BLOCKS,
        HCA_STATE_BLOCK_SIZE,
        COMPRESS_STATE_DIM,
        dtype=torch.float32,
    )
    prefix = int(metadata["prefix"])
    if prefix:
        logical_values = {
            position: (
                torch.rand(COMPRESS_STATE_DIM, generator=generator) - 0.5
            )
            * 0.05
            for position in range(max(0, prefix - COMPRESS_RATIO), prefix)
        }
        for rank in range(cp_size):
            flat = state[rank].view(-1, COMPRESS_STATE_DIM)
            for position, value in logical_values.items():
                row = _state_physical_row(state_tables[rank], position)
                if row >= 0:
                    flat[row] = value

    cmp_cache = torch.zeros(
        cp_size,
        CP_PREFILL_CMP_BLOCK_NUM,
        CMP_STORAGE_BLOCK_SIZE,
        1,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    completed_prefix = prefix // COMPRESS_RATIO
    if completed_prefix:
        logical_cmp = (
            torch.rand(completed_prefix, HEAD_DIM, generator=generator) - 0.5
        ).to(torch.bfloat16) * 0.1
        for rank in range(cp_size):
            flat = cmp_cache[rank].view(-1, HEAD_DIM)
            for slot in range(completed_prefix):
                row = _cmp_physical_row(
                    metadata["cmp_block_table"][rank], slot
                )
                if row >= 0:
                    flat[row] = logical_cmp[slot]

    specs.extend(
        [
            TensorSpec(
                "compress_state",
                list(state.shape),
                state.dtype,
                init_value=state,
            ),
            TensorSpec(
                "compress_state_block_table",
                list(state_tables.shape),
                state_tables.dtype,
                init_value=state_tables,
            ),
            TensorSpec(
                "kv_cache",
                list(kv_cache.shape),
                kv_cache.dtype,
                init_value=kv_cache,
            ),
            TensorSpec(
                "cmp_kv",
                list(cmp_cache.shape),
                cmp_cache.dtype,
                init_value=cmp_cache,
            ),
            TensorSpec(
                "cmp_block_table",
                list(metadata["cmp_block_table"].shape),
                metadata["cmp_block_table"].dtype,
                init_value=metadata["cmp_block_table"],
            ),
        ]
    )

    device_values = {
        "segment_starts_t": metadata["segment_starts"],
        "segment_active_lengths": raw_metadata["segment_active_lengths"],
        "owner_segments_t": metadata["owner_segments"],
        "predecessor_segments": raw_metadata["predecessor_segments"],
        "query_positions": raw_metadata["query_position_ids"],
        "query_requests": raw_metadata["query_token_to_request"],
        "overlay_positions": raw_metadata["overlay_position_ids"],
        "overlay_requests": raw_metadata["overlay_token_to_request"],
        "overlay_active_lengths": raw_metadata["overlay_active_lengths"],
        "swa_indices": raw_metadata["swa_indices"],
        "cmp_indices": metadata["cmp_indices"],
        "segment_tail_positions": metadata["segment_tail_positions"],
        "snapshot_positions": metadata["snapshot_positions"],
        "snapshot_valid": metadata["snapshot_valid"],
        "final_segment_t": metadata["final_segment_t"],
        "reverse_index": raw_metadata["reverse_index"],
        "owner_rank_table": metadata["owner_rank_table"],
        "owner_part_table": metadata["owner_part_table"],
        "final_win_seg_src": raw_metadata["final_win_seg_src"],
        "final_win_row_src": raw_metadata["final_win_row_src"],
        "final_slot_mapping": raw_metadata["final_slot_mapping"],
    }
    for name, value in device_values.items():
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )
    for name in tail_names:
        value = hca_values[name]
        specs.append(
            TensorSpec(
                name, list(value.shape), value.dtype, init_value=value
            )
        )
    specs.append(
        TensorSpec(
            "x_out",
            list(x_hc.shape),
            torch.float32,
        )
    )

    golden_prefill_cp_hca._ctx = {
        "cp_size": cp_size,
        "prefix": prefix,
        "lengths": [int(value) for value in metadata["segment_lengths"]],
        "starts": [int(value) for value in metadata["segment_starts"]],
        "owners": metadata["owner_segments"].tolist(),
        "final_segment": int(metadata["final_segment"]),
    }
    owner_spec = TensorSpec("cache_owner_rank_t", [cp_size, 1], torch.int32, init_value=0)
    head_position = next(i for i, spec in enumerate(specs) if spec.name == "attn_sink")
    specs.insert(head_position, owner_spec)
    return specs


def golden_prefill_cp_hca(tensors):
    """Compose CP-HCA golden outputs in logical-segment order."""
    import torch

    ctx = getattr(golden_prefill_cp_hca, "_ctx", None)
    if ctx is None:
        raise RuntimeError("CP-HCA golden context was not installed")
    cp_size = ctx["cp_size"]
    lengths = ctx["lengths"]
    owners = ctx["owners"]

    local_q = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        H,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_kv = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    local_norm = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        D,
        dtype=torch.bfloat16,
    )
    local_post = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT,
    )
    local_comb = torch.zeros(
        cp_size,
        LOCAL_PARTS,
        MAX_SEGMENT_TILES,
        TAIL_ROWS,
        HC_MULT * HC_MULT,
    )
    logical_hidden = torch.zeros(
        NUM_SEGMENTS, TAIL_ROWS, D, dtype=torch.bfloat16
    )
    logical_kv = torch.zeros(
        NUM_SEGMENTS, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16
    )

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 1]
                )
                x_tile = tensors["x_hc"][rank, part, tile]
                mixed = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                post = torch.zeros(TAIL_ROWS, HC_MULT)
                comb = torch.zeros(TAIL_ROWS, HC_MULT * HC_MULT)
                golden_hc_pre(
                    {
                        "x": x_tile,
                        "hc_fn": tensors["hc_attn_fn"],
                        "hc_scale": tensors["hc_attn_scale"],
                        "hc_base": tensors["hc_attn_base"],
                        "x_mixed": mixed,
                        "post": post,
                        "comb": comb,
                    }
                )
                normed = golden_rms_norm(mixed, tensors["attn_norm_w"])
                positions = tensors["query_positions"][rank, part, tile]
                rope_positions = positions.clamp_min(0).to(torch.long)
                q = torch.zeros(
                    TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16
                )
                kv = torch.zeros(
                    TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16
                )
                golden_qkv_proj_rope(
                    {
                        "x": normed,
                        "wq_a": tensors["wq_a"],
                        "wq_b": tensors["wq_b"],
                        "wq_b_scale": tensors["wq_b_scale"],
                        "wkv": tensors["wkv"],
                        "rope_cos": tensors["freqs_cos"].index_select(
                            0, rope_positions
                        ),
                        "rope_sin": tensors["freqs_sin"].index_select(
                            0, rope_positions
                        ),
                        "gamma_cq": tensors["gamma_cq"],
                        "gamma_ckv": tensors["gamma_ckv"],
                        "q": q,
                        "kv": kv,
                        "qr": torch.zeros(
                            TAIL_ROWS, Q_LORA, dtype=torch.int8
                        ),
                        "qr_scale": torch.zeros(TAIL_ROWS, 1),
                    }
                )
                local_q[rank, part, tile] = q
                local_kv[rank, part, tile] = kv
                local_norm[rank, part, tile] = normed
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb

            active_lengths = [
                int(tensors["overlay_active_lengths"][rank, part, tile, 1])
                for tile in range(MAX_SEGMENT_TILES)
            ]
            if any(active_lengths):
                hidden_rows = torch.cat(
                    [
                        local_norm[rank, part, tile, :active]
                        for tile, active in enumerate(active_lengths)
                        if active > 0
                    ],
                    dim=0,
                )
                kv_rows = torch.cat(
                    [
                        local_kv[rank, part, tile, :active]
                        for tile, active in enumerate(active_lengths)
                        if active > 0
                    ],
                    dim=0,
                )
                total = sum(active_lengths)
                valid = min(TAIL_ROWS, total)
                logical_hidden[segment, :valid] = hidden_rows[-valid:]
                logical_kv[segment, :valid] = kv_rows[-valid:]

    compressed_rows = []
    segment_snapshots = {}
    initial_state = tensors["compress_state"].clone()
    for segment in range(NUM_SEGMENTS):
        owner = cp_owner_rank(segment, cp_size)
        part = cp_owner_part(segment, cp_size)
        state_table = tensors["compress_state_block_table"][owner]
        scratch = torch.zeros_like(initial_state[owner])
        if segment == 0 and int(tensors["segment_starts_t"][0]) > 0:
            scratch.copy_(initial_state[owner])

        leaves = []
        predecessor = segment - 1
        if predecessor >= 0:
            pred_valid = min(TAIL_ROWS, lengths[predecessor])
            pred_x = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
            pred_positions = torch.zeros(TAIL_ROWS, dtype=torch.int32)
            if pred_valid:
                pred_x[:pred_valid] = logical_hidden[predecessor, :pred_valid]
                pred_positions[:pred_valid] = tensors[
                    "segment_tail_positions"
                ][predecessor, :pred_valid]
            leaves.append((pred_x, pred_positions, pred_valid, False))
        else:
            leaves.append(
                (
                    torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16),
                    torch.zeros(TAIL_ROWS, dtype=torch.int32),
                    0,
                    False,
                )
            )
        for tile in range(MAX_SEGMENT_TILES):
            active = int(
                tensors["overlay_active_lengths"][owner, part, tile, 1]
            )
            leaves.append(
                (
                    local_norm[owner, part, tile],
                    tensors["query_positions"][owner, part, tile],
                    active,
                    True,
                )
            )

        for leaf, (leaf_x, positions, active, publish) in enumerate(leaves):
            cmp_slots = torch.full(
                (TAIL_ROWS,), -1, dtype=torch.int64
            )
            state_slots = torch.full_like(cmp_slots, -1)
            logical_slot = -1
            for row in range(active):
                position = int(positions[row])
                state_slots[row] = _state_physical_row(
                    state_table, position
                )
                if publish and (position + 1) % COMPRESS_RATIO == 0:
                    cmp_slots[row] = 0
                    logical_slot = (position + 1) // COMPRESS_RATIO - 1
            leaf_cmp = torch.zeros(
                LEAF_CMP_BLOCKS,
                CMP_STORAGE_BLOCK_SIZE,
                1,
                HEAD_DIM,
                dtype=torch.bfloat16,
            )
            golden_prefill_compressor_ratio128(
                {
                    "x": leaf_x,
                    "compress_state": scratch,
                    "compress_state_block_table": state_table,
                    "wkv": tensors["cmp_wkv"],
                    "wgate": tensors["cmp_wgate"],
                    "ape": tensors["cmp_ape"],
                    "norm_w": tensors["cmp_norm_w"],
                    "freqs_cos": tensors["freqs_cos"],
                    "freqs_sin": tensors["freqs_sin"],
                    "cmp_kv": leaf_cmp,
                    "position_ids": positions,
                    "num_tokens": active,
                    "cmp_slot_mapping": cmp_slots,
                    "state_slot_mapping": state_slots,
                }
            )
            if logical_slot >= 0:
                compressed_rows.append(
                    (logical_slot, leaf_cmp.view(-1, HEAD_DIM)[0].clone())
                )

        valid = int(tensors["snapshot_valid"][owner, part])
        snapshot = torch.zeros(TAIL_ROWS, COMPRESS_STATE_DIM)
        scratch_flat = scratch.view(-1, COMPRESS_STATE_DIM)
        for row in range(valid):
            position = int(tensors["snapshot_positions"][owner, part, row])
            source = _state_physical_row(state_table, position)
            if source >= 0:
                snapshot[row] = scratch_flat[source]
        segment_snapshots[segment] = snapshot

    cmp_result = tensors["cmp_kv"].clone()
    for logical_slot, value in compressed_rows:
        for receiver in range(cp_size):
            destination = _cmp_physical_row(
                tensors["cmp_block_table"][receiver], logical_slot
            )
            if destination >= 0:
                cmp_result[receiver].view(-1, HEAD_DIM)[destination] = value

    state_result = initial_state.clone()
    final_segment = ctx["final_segment"]
    final_snapshot = segment_snapshots[final_segment]
    final_owner = cp_owner_rank(final_segment, cp_size)
    final_part = cp_owner_part(final_segment, cp_size)
    final_valid = int(tensors["snapshot_valid"][final_owner, final_part])
    for receiver in range(cp_size):
        state_flat = state_result[receiver].view(-1, COMPRESS_STATE_DIM)
        table = tensors["compress_state_block_table"][receiver]
        for row in range(final_valid):
            position = int(
                tensors["snapshot_positions"][final_owner, final_part, row]
            )
            destination = _state_physical_row(table, position)
            if destination >= 0:
                state_flat[destination] = final_snapshot[row]

    raw_initial = tensors["kv_cache"].clone()
    output = torch.zeros_like(tensors["x_out"])
    for rank in range(cp_size):
        persistent = raw_initial[rank].view(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            segment = owners[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 1]
                )
                fake = torch.zeros(
                    ORI_CACHE_ROWS + OVERLAY_ROWS,
                    HEAD_DIM,
                    dtype=torch.bfloat16,
                )
                fake[:persistent.shape[0]] = persistent
                predecessor = int(
                    tensors["predecessor_segments"][rank, part]
                )
                pred_valid = int(
                    tensors["overlay_active_lengths"][rank, part, tile, 0]
                )
                if pred_valid:
                    if tile == 0 and predecessor >= 0:
                        fake[
                            OVERLAY_BASE:OVERLAY_BASE + pred_valid
                        ] = logical_kv[predecessor, :pred_valid]
                    elif tile > 0:
                        fake[
                            OVERLAY_BASE:OVERLAY_BASE + pred_valid
                        ] = local_kv[rank, part, tile - 1, :pred_valid]
                fake[
                    OVERLAY_BASE + PRED_OVERLAY_ROWS:
                    OVERLAY_BASE + PRED_OVERLAY_ROWS + active
                ] = local_kv[rank, part, tile, :active]
                fake_cache = fake.view(
                    -1, BLOCK_SIZE, 1, HEAD_DIM
                )
                positions = tensors["query_positions"][rank, part, tile]
                rope_positions = positions.clamp_min(0).to(torch.long)
                attn = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                golden_prefill_sparse_attn(
                    {
                        "q": local_q[rank, part, tile],
                        "ori_kv": fake_cache,
                        "swa_indices": tensors["swa_indices"][
                            rank, part, tile
                        ],
                        "cmp_kv": cmp_result[rank],
                        "cmp_block_table": tensors["cmp_block_table"][rank],
                        "cmp_storage_block_size": CMP_STORAGE_BLOCK_SIZE,
                        "cmp_indices": tensors["cmp_indices"][
                            rank, part, tile
                        ],
                        "attn_sink": tensors["attn_sink"],
                        "num_tokens": active,
                        "freqs_cos": tensors["freqs_cos"].index_select(
                            0, rope_positions
                        ),
                        "freqs_sin": tensors["freqs_sin"].index_select(
                            0, rope_positions
                        ),
                        "wo_a": tensors["wo_a"],
                        "wo_b": tensors["wo_b"],
                        "wo_b_scale": tensors["wo_b_scale"],
                        "attn_out": attn,
                    }
                )
                y = torch.zeros(TAIL_ROWS, HC_MULT, D)
                golden_hc_post_prefill(
                    {
                        "x": attn,
                        "residual": tensors["x_hc"][rank, part, tile],
                        "post": local_post[rank, part, tile],
                        "comb": local_comb[rank, part, tile],
                        "y": y,
                        "num_tokens": active,
                    }
                )
                output[rank, part, tile] = y

    raw_result = raw_initial.clone().view(cp_size, -1, HEAD_DIM)
    for row in range(TAIL_ROWS):
        segment = int(tensors["final_win_seg_src"][row])
        source_row = int(tensors["final_win_row_src"][row])
        destination = int(tensors["final_slot_mapping"][row])
        if segment >= 0 and source_row >= 0 and destination >= 0:
            raw_result[:, destination] = logical_kv[segment, source_row]

    for receiver in range(cp_size):
        if receiver == int(tensors["cache_owner_rank_t"][receiver, 0]):
            tensors["compress_state"][receiver] = state_result[receiver]
            tensors["cmp_kv"][receiver] = cmp_result[receiver]
            tensors["kv_cache"][receiver] = raw_result.view_as(tensors["kv_cache"])[receiver]
    tensors["x_out"][:] = output


if __name__ == "__main__" and not _run_cp_fixture:
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 packed prefill HCA correctness test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of S=WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token count (q_len), capped by T; passed to the kernel as num_tokens.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    compare_tokens = args.num_tokens

    result = run(
        fn=prefill_attention_hca_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
        ),
        golden_fn=golden_prefill_attention_hca,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-2,
        atol=1e-2,
        compile_only=args.compile_only,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1,
                                   valid_rows=compare_tokens, zero_tail=True),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__" and _run_cp_fixture:
    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 context-parallel HCA test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=list(CP_CHOICES))
    parser.add_argument("--num-tokens", type=int, default=None, help="actual request length; defaults to full capacity")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--enable-chip-swimlane", action="store_true")
    args = parser.parse_args()

    from golden import ratio_allclose, ratio_reldiff, run

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")
    result = run(
        fn=prefill_cp_hca_test,
        specs=build_cp_tensor_specs(args.cp, num_tokens=args.num_tokens),
        golden_fn=golden_prefill_cp_hca,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[: args.cp], num_sub_workers=0
            ),
            dump_passes=args.dump_passes,
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            ring_heap=PREFILL_CP_HCA_RING_HEAP,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

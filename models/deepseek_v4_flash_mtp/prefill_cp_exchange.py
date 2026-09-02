# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Context-parallel prefill tail exchange, compact-cache exchange, and sparse-source staging."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    FLASH as M,
    HCA_STATE_PHYSICAL_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
)
from lm_head import (
    GROUP_LOGIT_ROWS as LM_HEAD_GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS as LM_HEAD_MAX_LOGIT_ROWS,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP as LM_HEAD_VOCAB_PER_TP,
    lm_head,
)
from prefill_compressor_ratio128 import (
    CMP_STORAGE_BLOCK_SIZE as HCA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    MAX_SEQ_LEN,
)
from prefill_compressor_ratio4 import (
    CMP_STORAGE_BLOCK_SIZE as CSA_CMP_STORAGE_BLOCK_SIZE,
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM as MAIN_STATE_DIM,
    CSA_STATE_BLOCK_SIZE as MAIN_STATE_BLOCK_SIZE,
    HEAD_DIM as MAIN_HEAD_DIM,
)
from prefill_cp_zigzag import (
    CP_SIZE,
    CP_PREFILL_CMP_BLOCK_NUM as PREFILL_CMP_BLOCK_NUM,
    CP_TAIL_WINDOW_ROWS,
    EPOCHS,
    HEAD_DIM,
    MAX_SEGMENT_TILES,
    NUM_SEGMENTS,
    ROW_TILE,
    TAIL_ROWS,
)
from prefill_indexer_compressor import (
    COMPRESS_STATE_DIM as INNER_STATE_DIM,
    HEAD_DIM as INNER_HEAD_DIM,
    INNER_STATE_BLOCK_SIZE,
)
from prefill_sparse_attn import (
    PREFILL_SPARSE_PAD,
)

CP_CMP_BLOCK_NUM_DYN = pl.dynamic("CP_CMP_BLOCK_NUM_DYN")
CP_CMP_STORAGE_BLOCK_SIZE_DYN = pl.dynamic("CP_CMP_STORAGE_BLOCK_SIZE_DYN")


# model config
D = M.hidden_size
WIN = M.sliding_window
IDX_TOPK = M.index_topk

# CP exchange layout
LOCAL_PARTS = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD
ORI_CACHE_ROWS = PREFILL_ORI_MAX_BLOCKS * BLOCK_SIZE
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = 2 * TAIL_ROWS
OVERLAY_SOURCES = 2

CMP_ROWS_PER_SEGMENT = (
    MAX_SEGMENT_TILES * TAIL_ROWS // HCA_COMPRESS_RATIO
)
CMP_ROWS_PER_RANK = LOCAL_PARTS * CMP_ROWS_PER_SEGMENT
CMP_META_DIM = 8
STATE_META_DIM = 8
CMP_WINDOW_ROWS = CP_SIZE * CMP_ROWS_PER_RANK
STATE_WINDOW_ROWS = CP_SIZE * TAIL_ROWS

ROWS_PER_RANK = (
    LOCAL_PARTS * MAX_SEGMENT_TILES * TAIL_ROWS // CSA_COMPRESS_RATIO
)
STATE_ROWS_PER_RANK = 8
META_DIM = 8
RECORDS_PER_WINDOW = CP_SIZE * ROWS_PER_RANK
STATE_RECORDS_PER_WINDOW = CP_SIZE * STATE_ROWS_PER_RANK
# FP16 scale rows must remain 32-byte aligned on PTOAS 0.60.
SCALE_TILE_COLS = 16
MAIN_CACHE_ROWS = PREFILL_CMP_BLOCK_NUM * CSA_CMP_STORAGE_BLOCK_SIZE
MAIN_STATE_ROWS = CSA_STATE_PHYSICAL_BLOCKS * MAIN_STATE_BLOCK_SIZE
INNER_STATE_ROWS = CSA_INNER_STATE_PHYSICAL_BLOCKS * INNER_STATE_BLOCK_SIZE
CP_LAST_HIDDEN_EPOCH = 1


@pl.jit(auto_scope=False)
def prefill_cp_last_hidden_lm_head(
    hidden_states: pl.Tensor[[LOCAL_ROWS, D], pl.BF16],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    lm_head_weight: pl.Tensor[[LM_HEAD_VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[LM_HEAD_MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[
        pl.Tensor[[LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]
    ],
    cp_hidden_window: pld.DistributedTensor[[1, D], pl.BF16],
    cp_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cp_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    lm_hidden_window: pld.DistributedTensor[
        [LM_HEAD_GROUP_LOGIT_ROWS, D], pl.BF16
    ],
    lm_hidden_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    lm_logits_window: pld.DistributedTensor[
        [LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32
    ],
    lm_logits_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    my_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]:
    """Feed the Recipes CP-global final hidden into the existing TP LM head.

    CANN Recipes writes the global-final row on the unique owner, writes zero
    on every other CP rank, and SUM-allreduces the resulting ``[1, D]`` row.
    Under that one-owner invariant, the push broadcast below is numerically
    identical.  It stays in this JIT wrapper because PTOAS 0.60 cannot map an
    inline callee with the three distributed InOut windows plus one Out tensor
    to an explicit return parameter.

    ``cp_ready``/``cp_consumed`` form a two-phase reuse handshake: receivers
    acknowledge only after copying their local window into ``last_hidden``,
    and the owner waits for all acknowledgements before clearing them.
    """
    last_hidden = pl.create_tensor([1, D], dtype=pl.BF16)
    final_segment = pl.read(final_segment_t, [0])
    final_owner = pl.read(owner_rank_table, [final_segment])
    final_part = pl.read(owner_part_table, [final_segment])
    epoch = pl.cast(CP_LAST_HIDDEN_EPOCH, pl.INT32)
    zero = pl.cast(0, pl.INT32)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_select",
    ) as select_tid:
        last_hidden[0:1, 0:D] = pl.full(
            [1, D], dtype=pl.BF16, value=0.0
        )
        if my_rank == final_owner:
            active = pl.read(segment_active_lengths, [final_part])
            source_row_raw = (
                final_part
                * pl.cast(MAX_SEGMENT_TILES * TAIL_ROWS, pl.INT32)
                + active
                - pl.cast(1, pl.INT32)
            )
            source_row = pl.cast(source_row_raw, target_type=pl.INDEX)
            last_hidden[0:1, 0:D] = hidden_states[
                source_row:source_row + 1, 0:D
            ]

    # PTOAS 0.60 requires each generated orchestration scope to have at most
    # one implicit Out/InOut result.  Keep payload, ready, consumed, and the
    # GM readback in separate tasks, with explicit edges carrying the Recipes
    # two-phase reuse protocol.
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_put",
        deps=[select_tid],
    ) as put_tid:
        if my_rank == final_owner:
            for peer in pl.range(CP_SIZE):
                if peer != my_rank:
                    pld.tensor.put(
                        dst=cp_hidden_window,
                        peer=peer,
                        src=last_hidden,
                        dst_offsets=[0, 0],
                        src_offsets=[0, 0],
                        shape=[1, D],
                    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_ready",
        deps=[put_tid],
    ) as ready_tid:
        if my_rank == final_owner:
            for peer in pl.range(CP_SIZE):
                if peer != my_rank:
                    pld.system.notify(
                        target=cp_ready,
                        peer=peer,
                        offsets=[my_rank, 0],
                        value=epoch,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_ready_wait",
        deps=[select_tid],
    ) as ready_wait_tid:
        if my_rank != final_owner:
            pld.system.wait(
                signal=cp_ready,
                offsets=[final_owner, 0],
                expected=epoch,
                cmp=pld.WaitCmp.Ge,
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_readback",
        deps=[select_tid, ready_wait_tid],
    ) as readback_tid:
        if my_rank != final_owner:
            last_hidden[0:1, 0:D] = cp_hidden_window[0:1, 0:D]

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_ready_clear",
        deps=[readback_tid],
    ) as ready_clear_tid:
        if my_rank != final_owner:
            pl.write(cp_ready, [final_owner, 0], zero)

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_consumed",
        deps=[ready_clear_tid],
    ) as consumed_tid:
        if my_rank != final_owner:
            pld.system.notify(
                target=cp_consumed,
                peer=final_owner,
                offsets=[my_rank, 0],
                value=epoch,
                op=pld.NotifyOp.AtomicAdd,
            )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_consumed_wait",
        deps=[ready_tid],
    ) as consumed_wait_tid:
        if my_rank == final_owner:
            for peer in pl.range(CP_SIZE):
                if peer != my_rank:
                    pld.system.wait(
                        signal=cp_consumed,
                        offsets=[peer, 0],
                        expected=epoch,
                        cmp=pld.WaitCmp.Ge,
                    )

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="prefill_cp_last_hidden_consumed_clear",
        deps=[consumed_tid, consumed_wait_tid],
    ):
        if my_rank == final_owner:
            for peer in pl.range(CP_SIZE):
                if peer != my_rank:
                    pl.write(cp_consumed, [peer, 0], zero)
    lm_head(
        last_hidden,
        lm_head_weight,
        logit_row_indices,
        logits,
        lm_hidden_window,
        lm_hidden_done,
        lm_logits_window,
        lm_logits_done,
        group_base,
        tp_rank,
        done_epoch,
    )
    return logits


@pl.jit.inline
def _prefill_cp_hidden_tail_exchange_wave(
    local_hidden_tail: pl.Tensor[
        [EPOCHS * LOCAL_PARTS * TAIL_ROWS, D], pl.BF16
    ],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    hidden_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    logical_hidden_out: pl.Out[
        pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]
    ],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[EPOCHS * CP_TAIL_WINDOW_ROWS, D], pl.BF16]:
    """Exchange only hidden tails, matching the Recipes CP prefill contract.

    The ready/consumed protocol intentionally matches the dual-tail wave one
    for one so callers can keep using the same monotonic cross-layer epoch.
    Projected KV is produced locally after the hidden exchange.
    """
    epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_epoch, cmp=pld.WaitCmp.Ge,
            )

    for peer in pl.range(CP_SIZE):
        for part in pl.range(LOCAL_PARTS):
            publish_pos = my_rank * LOCAL_PARTS + part
            publish_dst_row = publish_pos * TAIL_ROWS
            src_row_base = (
                payload_epoch * LOCAL_PARTS * TAIL_ROWS
                + part * TAIL_ROWS
            )
            pld.tensor.put(
                dst=hidden_window,
                peer=peer,
                src=local_hidden_tail,
                dst_offsets=[publish_dst_row, 0],
                src_offsets=[src_row_base, 0],
                shape=[TAIL_ROWS, D],
                chunk_rows=ROW_TILE,
                chunk_cols=D,
                pipeline=True,
            )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    for seg in pl.range(NUM_SEGMENTS):
        gather_pos = reverse_index[seg]
        owner = owner_rank_table[seg]
        if owner != my_rank:
            pld.system.wait(
                signal=ready, offsets=[owner, 0],
                expected=epoch_value, cmp=pld.WaitCmp.Ge,
            )
        gather_src_row = gather_pos * TAIL_ROWS
        gather_dst_row = (
            payload_epoch * CP_TAIL_WINDOW_ROWS + seg * TAIL_ROWS
        )
        for t0 in pl.range(0, TAIL_ROWS, ROW_TILE):
            hidden_tile = hidden_window[
                gather_src_row + t0:gather_src_row + t0 + ROW_TILE,
                0:D,
            ]
            logical_hidden_out[
                gather_dst_row + t0:gather_dst_row + t0 + ROW_TILE,
                0:D,
            ] = hidden_tile

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    return logical_hidden_out


@pl.jit.inline
def _prefill_cp_hca_compact_exchange_commit_wave(
    local_cmp_payload: pl.Tensor[
        [EPOCHS * CMP_ROWS_PER_RANK, HEAD_DIM], pl.BF16
    ],
    local_cmp_meta: pl.Tensor[
        [EPOCHS * CMP_ROWS_PER_RANK, CMP_META_DIM], pl.INT32
    ],
    local_state_payload: pl.Tensor[
        [EPOCHS * TAIL_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    local_state_meta: pl.Tensor[[EPOCHS, STATE_META_DIM], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
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
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM * HCA_CMP_STORAGE_BLOCK_SIZE, HEAD_DIM], pl.BF16
        ]
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_PHYSICAL_BLOCKS * HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
) -> None:
    """Publish HCA compact rows and commit receiver-local cache/state.

    ``payload_epoch`` selects rows in the invocation-local payload tensors
    (``local_cmp_payload``/``local_cmp_meta``/``local_state_payload``/
    ``local_state_meta``); ``comm_epoch`` drives the HCA compact
    ready/consumed counters (``consumed >= comm_epoch``,
    ``ready >= comm_epoch + 1``).
    """
    epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_epoch, cmp=pld.WaitCmp.Ge,
            )

    cmp_src_row = payload_epoch * CMP_ROWS_PER_RANK
    state_src_row = payload_epoch * TAIL_ROWS
    for peer in pl.range(CP_SIZE):
        cmp_dst_row = my_rank * CMP_ROWS_PER_RANK
        state_dst_row = my_rank * TAIL_ROWS
        pld.tensor.put(
            dst=cmp_window, peer=peer, src=local_cmp_payload,
            dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, HEAD_DIM],
            chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=cmp_meta_window, peer=peer, src=local_cmp_meta,
            dst_offsets=[cmp_dst_row, 0], src_offsets=[cmp_src_row, 0], shape=[CMP_ROWS_PER_RANK, CMP_META_DIM],
            chunk_rows=CMP_ROWS_PER_RANK, chunk_cols=CMP_META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=state_window, peer=peer, src=local_state_payload,
            dst_offsets=[state_dst_row, 0], src_offsets=[state_src_row, 0], shape=[TAIL_ROWS, COMPRESS_STATE_DIM],
            chunk_rows=ROW_TILE, chunk_cols=COMPRESS_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=state_meta_window, peer=peer, src=local_state_meta,
            dst_offsets=[my_rank, 0], src_offsets=[payload_epoch, 0], shape=[1, STATE_META_DIM],
            chunk_rows=1, chunk_cols=STATE_META_DIM, pipeline=True,
        )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=ready, offsets=[peer, 0],
                expected=epoch_value, cmp=pld.WaitCmp.Ge,
            )

    for segment in pl.range(NUM_SEGMENTS):
        cmp_owner = owner_rank_table[segment]
        owner_part = owner_part_table[segment]
        for row in pl.range(CMP_ROWS_PER_SEGMENT):
            cmp_source_row = cmp_owner * CMP_ROWS_PER_RANK + owner_part * CMP_ROWS_PER_SEGMENT + row
            valid = pl.read(cmp_meta_window, [cmp_source_row, 0])
            meta_segment = pl.read(cmp_meta_window, [cmp_source_row, 1])
            logical_slot = pl.read(cmp_meta_window, [cmp_source_row, 3])
            if valid > 0:
                if meta_segment == segment:
                    if logical_slot >= 0:
                        logical_block = pl.cast(logical_slot // HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                        if logical_block < PREFILL_CMP_MAX_BLOCKS:
                            physical_block = pl.read(cmp_block_table, [logical_block])
                            if physical_block >= 0:
                                intra = pl.cast(logical_slot % HCA_CMP_STORAGE_BLOCK_SIZE, pl.INDEX)
                                cmp_row_tile = cmp_window[
                                    cmp_source_row : cmp_source_row + 1,
                                    0:HEAD_DIM,
                                ]
                                cache_row = (
                                    pl.cast(physical_block, pl.INDEX)
                                    * HCA_CMP_STORAGE_BLOCK_SIZE
                                    + intra
                                )
                                cmp_kv[
                                    cache_row : cache_row + 1, 0:HEAD_DIM
                                ] = cmp_row_tile

    for state_owner in pl.range(CP_SIZE):
        state_valid = pl.read(state_meta_window, [state_owner, 0])
        valid_rows = pl.read(state_meta_window, [state_owner, 2])
        end_position = pl.read(state_meta_window, [state_owner, 3])
        if state_valid > 0:
            for row in pl.range(TAIL_ROWS):
                if row < valid_rows:
                    absolute_position = end_position - valid_rows + row
                    if absolute_position >= 0:
                        if absolute_position < MAX_SEQ_LEN:
                            logical_block = pl.cast(absolute_position // HCA_STATE_BLOCK_SIZE, pl.INDEX)
                            if logical_block < HCA_STATE_MAX_BLOCKS:
                                physical_block = pl.read(compress_state_block_table, [logical_block])
                                if physical_block >= 0:
                                    intra = pl.cast(absolute_position % HCA_STATE_BLOCK_SIZE, pl.INDEX)
                                    state_source_row = state_owner * TAIL_ROWS + row
                                    state_row_tile = pl.slice(
                                        state_window,
                                        [1, COMPRESS_STATE_DIM],
                                        [state_source_row, 0],
                                    )
                                    state_row = pl.cast(physical_block, pl.INDEX) * HCA_STATE_BLOCK_SIZE + intra
                                    compress_state[state_row : state_row + 1, 0:COMPRESS_STATE_DIM] = state_row_tile

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    # ``cmp_kv`` is caller-owned InOut storage.  Do not return a scoped SSA
    # alias: PyPTO 0.60 would classify the compact task argument as Out and
    # discard untouched cache rows instead of preserving their input values.


@pl.jit.inline
def _prefill_cp_csa_compact_transport_wave(
    main_payload: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, MAIN_HEAD_DIM], pl.BF16
    ],
    idx_payload: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, INNER_HEAD_DIM], pl.INT8
    ],
    idx_scale: pl.Tensor[
        [EPOCHS * ROWS_PER_RANK, SCALE_TILE_COLS], pl.FP16
    ],
    record_meta: pl.Tensor[[EPOCHS * ROWS_PER_RANK, META_DIM], pl.INT32],
    main_state_payload: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, MAIN_STATE_DIM], pl.FP32
    ],
    inner_state_payload: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, INNER_STATE_DIM], pl.FP32
    ],
    main_state_meta: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32
    ],
    inner_state_meta: pl.Tensor[
        [EPOCHS * STATE_ROWS_PER_RANK, STATE_META_DIM], pl.INT32
    ],
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, MAIN_HEAD_DIM], pl.BF16
    ],
    idx_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, INNER_HEAD_DIM], pl.INT8
    ],
    scale_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP16
    ],
    record_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, META_DIM], pl.INT32
    ],
    main_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    payload_epoch: pl.Scalar[pl.INT32],
    comm_epoch: pl.Scalar[pl.INT32],
):
    comm_i32 = pl.cast(comm_epoch, pl.INT32)
    ready_expected = pl.cast(comm_i32 + 1, pl.INT32)
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=consumed, offsets=[peer, 0],
                expected=comm_i32, cmp=pld.WaitCmp.Ge,
            )

    payload_row = payload_epoch * ROWS_PER_RANK
    state_row = payload_epoch * STATE_ROWS_PER_RANK
    destination_row = my_rank * ROWS_PER_RANK
    destination_state_row = my_rank * STATE_ROWS_PER_RANK
    for peer in pl.range(CP_SIZE):
        pld.tensor.put(
            dst=main_window, peer=peer, src=main_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, MAIN_HEAD_DIM],
            chunk_rows=8, chunk_cols=MAIN_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=idx_window, peer=peer, src=idx_payload,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, INNER_HEAD_DIM],
            chunk_rows=8, chunk_cols=INNER_HEAD_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=scale_window, peer=peer, src=idx_scale,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, SCALE_TILE_COLS],
            chunk_rows=8, chunk_cols=SCALE_TILE_COLS, pipeline=True,
        )
        pld.tensor.put(
            dst=record_window, peer=peer, src=record_meta,
            dst_offsets=[destination_row, 0], src_offsets=[payload_row, 0], shape=[ROWS_PER_RANK, META_DIM],
            chunk_rows=8, chunk_cols=META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_window, peer=peer, src=main_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, MAIN_STATE_DIM],
            chunk_rows=4, chunk_cols=MAIN_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=main_state_meta_window, peer=peer, src=main_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_window, peer=peer, src=inner_state_payload,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, INNER_STATE_DIM],
            chunk_rows=4, chunk_cols=INNER_STATE_DIM, pipeline=True,
        )
        pld.tensor.put(
            dst=inner_state_meta_window, peer=peer, src=inner_state_meta,
            dst_offsets=[destination_state_row, 0], src_offsets=[state_row, 0],
            shape=[STATE_ROWS_PER_RANK, STATE_META_DIM],
            chunk_rows=4, chunk_cols=STATE_META_DIM, pipeline=True,
        )

    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=ready, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.wait(
                signal=ready, offsets=[peer, 0],
                expected=ready_expected, cmp=pld.WaitCmp.Ge,
            )


@pl.jit.inline
def _prefill_cp_csa_compact_finish_wave(
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    for peer in pl.range(CP_SIZE):
        if peer != my_rank:
            pld.system.notify(
                target=consumed, peer=peer, offsets=[my_rank, 0],
                value=1, op=pld.NotifyOp.AtomicAdd,
            )



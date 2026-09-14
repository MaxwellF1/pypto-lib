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

"""Exact CP request transfer, owner output isolation and communication reuse tests."""

import argparse

import torch

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from golden import TensorSpec, run
from prefill_cp_exchange import (
    CP_REQUEST_CAPACITY,
    CP_REQUEST_COPY_COLS,
    CP_REQUEST_HC_DIM,
    CP_REQUEST_INNER_TABLE_COLS,
    CP_REQUEST_MAIN_TABLE_COLS,
    CP_REQUEST_TABLE_COLS,
    CP_REQUEST_TOKENS_DYN,
    CP_SIZE,
    D,
    HCA_STATE_MAX_BLOCKS,
    LOCAL_ROWS,
    MAX_SEGMENT_TILES,
    NUM_SEGMENTS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_MAX_BLOCKS,
    ROW_TILE,
    TAIL_ROWS,
    _prefill_cp_request_header as request_header,
    _prefill_cp_scatter_request as scatter_request,
    _prefill_cp_gather_hidden as gather_request,
    _prefill_cp_release_request as release_request,
)

TOKENS = CP_REQUEST_CAPACITY
TABLES = (
    "ori_block_table",
    "hca_cmp_block_table",
    "csa_cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
)
TABLE_COLS = (
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_CMP_MAX_BLOCKS,
    HCA_STATE_MAX_BLOCKS,
    CP_REQUEST_MAIN_TABLE_COLS,
    CP_REQUEST_INNER_TABLE_COLS,
)


@pl.jit(auto_scope=False)
def request_rank(
    num_tokens_per_owner: pl.Tensor[[CP_SIZE], pl.INT32],
    position_ids: pl.Tensor[[CP_REQUEST_TOKENS_DYN], pl.INT32],
    x_hc: pl.Tensor[[CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    input_ids: pl.Tensor[[1, CP_REQUEST_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[1, PREFILL_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[1, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32],
    header_window: pld.DistributedTensor[[1, 16], pl.INT32],
    input_window: pld.DistributedTensor[[LOCAL_ROWS, CP_REQUEST_HC_DIM], pl.FP32],
    ids_window: pld.DistributedTensor[[1, LOCAL_ROWS * 2], pl.INT32],
    tables_window: pld.DistributedTensor[[7, CP_REQUEST_TABLE_COLS], pl.INT32],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hidden_window: pld.DistributedTensor[[CP_REQUEST_CAPACITY, D], pl.BF16],
    tail_window: pld.DistributedTensor[[NUM_SEGMENTS * TAIL_ROWS, CP_REQUEST_HC_DIM], pl.FP32],
    complete: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    released: pld.DistributedTensor[[CP_SIZE, 16], pl.INT32],
    hidden_out: pl.InOut[pl.Tensor[[CP_REQUEST_TOKENS_DYN, D], pl.BF16]],
    tail_out: pl.InOut[pl.Tensor[[TAIL_ROWS, CP_REQUEST_HC_DIM], pl.FP32]],
    audit: pl.InOut[pl.Tensor[[CP_SIZE, 9, CP_REQUEST_TABLE_COLS], pl.INT32]],
    request_owner: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    header = pl.create_tensor([1, 16], dtype=pl.INT32)
    if request_owner == 0:
        for row in pl.spmd(CP_SIZE * 9):
            for col in pl.range(CP_REQUEST_TABLE_COLS // 128):
                zeros = pl.tile.full([1, 1, 128], value=0, dtype=pl.INT32)
                pl.store(zeros, [row // 9, row % 9, col * 128], audit)
        for row in pl.spmd(TOKENS // ROW_TILE):
            for col in pl.range(D // CP_REQUEST_COPY_COLS):
                zero = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.BF16)
                pl.store(zero, [row * ROW_TILE, col * CP_REQUEST_COPY_COLS], hidden_out)
        for row in pl.spmd(TAIL_ROWS // ROW_TILE):
            for col in pl.range(CP_REQUEST_HC_DIM // CP_REQUEST_COPY_COLS):
                zero_tail = pl.tile.full([ROW_TILE, CP_REQUEST_COPY_COLS], value=0.0, dtype=pl.FP32)
                pl.store(zero_tail, [row * ROW_TILE, col * CP_REQUEST_COPY_COLS], tail_out)
    if pl.read(num_tokens_per_owner, [request_owner]) > 0:
        control = pl.create_tensor([1, 16], dtype=pl.INT32)
        local_x_hc = pl.create_tensor([LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
        local_input_ids = pl.create_tensor([1, LOCAL_ROWS], dtype=pl.INT64)
        local_ori_block_table = pl.create_tensor([1, PREFILL_ORI_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_cmp_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_cmp_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_idx_block_table = pl.create_tensor([1, PREFILL_CMP_MAX_BLOCKS], dtype=pl.INT32)
        local_hca_compress_state_block_table = pl.create_tensor([1, HCA_STATE_MAX_BLOCKS], dtype=pl.INT32)
        local_csa_compress_state_block_table = pl.create_tensor([1, CP_REQUEST_MAIN_TABLE_COLS], dtype=pl.INT32)
        local_csa_inner_compress_state_block_table = pl.create_tensor([1, CP_REQUEST_INNER_TABLE_COLS], dtype=pl.INT32)
        request_header(num_tokens_per_owner, position_ids, header, request_owner, my_rank)
        (
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
        ) = scatter_request(
            header, x_hc, input_ids,
            ori_block_table, hca_cmp_block_table, csa_cmp_block_table, idx_block_table,
            hca_compress_state_block_table, csa_compress_state_block_table, csa_inner_compress_state_block_table,
            header_window, input_window, ids_window, tables_window, ready,
            control, local_x_hc, local_input_ids,
            local_ori_block_table, local_hca_cmp_block_table, local_csa_cmp_block_table, local_idx_block_table,
            local_hca_compress_state_block_table, local_csa_compress_state_block_table, local_csa_inner_compress_state_block_table,
            my_rank,
        )
        ids_flat = pl.reshape(local_input_ids, [1, 1, LOCAL_ROWS])
        metadata = pl.reshape(control, [1, 1, 16])
        table_2 = pl.reshape(local_ori_block_table, [1, 1, PREFILL_ORI_MAX_BLOCKS])
        table_3 = pl.reshape(local_hca_cmp_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_4 = pl.reshape(local_csa_cmp_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_5 = pl.reshape(local_idx_block_table, [1, 1, PREFILL_CMP_MAX_BLOCKS])
        table_6 = pl.reshape(local_hca_compress_state_block_table, [1, 1, HCA_STATE_MAX_BLOCKS])
        table_7 = pl.reshape(local_csa_compress_state_block_table, [1, 1, CP_REQUEST_MAIN_TABLE_COLS])
        table_8 = pl.reshape(local_csa_inner_compress_state_block_table, [1, 1, CP_REQUEST_INNER_TABLE_COLS])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="capture_request_metadata"):
            for col in pl.range(LOCAL_ROWS * 2 // 128):
                ids = pl.load(ids_flat, [0, 0, col * 64], [1, 1, 64])
                ids_words = pl.reinterpret_view(ids, pl.INT32)
                pl.store(ids_words, [request_owner, 0, col * 128], audit)
            control_values = pl.load(metadata, [0, 0, 0], [1, 1, 16])
            pl.store(control_values, [request_owner, 1, 0], audit)
            for col in pl.range(PREFILL_ORI_MAX_BLOCKS // 128):
                values_2 = pl.load(table_2, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_2, [request_owner, 2, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_3 = pl.load(table_3, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_3, [request_owner, 3, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_4 = pl.load(table_4, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_4, [request_owner, 4, col * 128], audit)
            for col in pl.range(PREFILL_CMP_MAX_BLOCKS // 128):
                values_5 = pl.load(table_5, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_5, [request_owner, 5, col * 128], audit)
            for col in pl.range(HCA_STATE_MAX_BLOCKS // 128):
                values_6 = pl.load(table_6, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_6, [request_owner, 6, col * 128], audit)
            for col in pl.range(CP_REQUEST_MAIN_TABLE_COLS // 128):
                values_7 = pl.load(table_7, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_7, [request_owner, 7, col * 128], audit)
            for col in pl.range(CP_REQUEST_INNER_TABLE_COLS // 128):
                values_8 = pl.load(table_8, [0, 0, col * 128], [1, 1, 128])
                pl.store(values_8, [request_owner, 8, col * 128], audit)
        local_hidden = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
        for block in pl.spmd(LOCAL_ROWS // ROW_TILE):
            for col in pl.range(D // CP_REQUEST_COPY_COLS):
                values = pl.load(local_x_hc, [block * ROW_TILE, col * CP_REQUEST_COPY_COLS], [ROW_TILE, CP_REQUEST_COPY_COLS])
                rounded = pl.cast(values, pl.BF16, mode="rint")
                pl.store(rounded, [block * ROW_TILE, col * CP_REQUEST_COPY_COLS], local_hidden)
        gather_request(
            control, local_hidden, local_x_hc,
            hidden_window, tail_window, complete,
            hidden_out, tail_out, my_rank,
        )
    release_request(
        hidden_out, tail_out, released,
        ready, ready, ready, ready, ready, ready, ready, ready, ready, ready,
        my_rank,
    )
    return hidden_out, tail_out


@pl.jit.host
def request_host(
    num_tokens_per_owner: pl.Tensor[[CP_SIZE], pl.INT32],
    position_ids: pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN], pl.INT32],
    x_hc: pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN, CP_REQUEST_HC_DIM], pl.FP32],
    input_ids: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_TOKENS_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_ORI_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[CP_SIZE, 1, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    hca_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_MAIN_TABLE_COLS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[CP_SIZE, 1, CP_REQUEST_INNER_TABLE_COLS], pl.INT32],
    hidden_out: pl.Out[pl.Tensor[[CP_SIZE, CP_REQUEST_TOKENS_DYN, D], pl.BF16]],
    tail_out: pl.Out[pl.Tensor[[CP_SIZE, TAIL_ROWS, CP_REQUEST_HC_DIM], pl.FP32]],
    audit: pl.Out[pl.Tensor[[CP_SIZE, CP_SIZE, 9, CP_REQUEST_TABLE_COLS], pl.INT32]],
):
    x_hc.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    input_ids.bind_dynamic(2, CP_REQUEST_TOKENS_DYN)
    position_ids.bind_dynamic(1, CP_REQUEST_TOKENS_DYN)
    header_window_buf = pld.alloc_window_buffer([1, 16], dtype=pl.INT32)
    input_window_buf = pld.alloc_window_buffer([LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
    ids_window_buf = pld.alloc_window_buffer([1, LOCAL_ROWS * 2], dtype=pl.INT32)
    tables_window_buf = pld.alloc_window_buffer([7, CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    hidden_window_buf = pld.alloc_window_buffer([CP_REQUEST_CAPACITY, D], dtype=pl.BF16)
    tail_window_buf = pld.alloc_window_buffer([NUM_SEGMENTS * TAIL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
    complete_buf = pld.alloc_window_buffer([CP_SIZE, 16], dtype=pl.INT32)
    released_buf = pld.alloc_window_buffer([CP_SIZE, 16], dtype=pl.INT32)
    for repetition in pl.range(3):
        for request_owner in pl.range(CP_SIZE):
            for rank in pl.range(CP_SIZE):
                hidden_window = pld.window(hidden_window_buf, [CP_REQUEST_CAPACITY, D], dtype=pl.BF16)
                tail_window = pld.window(tail_window_buf, [NUM_SEGMENTS * TAIL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
                complete = pld.window(complete_buf, [CP_SIZE, 1], dtype=pl.INT32)
                released = pld.window(released_buf, [CP_SIZE, 16], dtype=pl.INT32)
                header_window = pld.window(header_window_buf, [1, 16], dtype=pl.INT32)
                input_window = pld.window(input_window_buf, [LOCAL_ROWS, CP_REQUEST_HC_DIM], dtype=pl.FP32)
                ids_window = pld.window(ids_window_buf, [1, LOCAL_ROWS * 2], dtype=pl.INT32)
                tables_window = pld.window(tables_window_buf, [7, CP_REQUEST_TABLE_COLS], dtype=pl.INT32)
                ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
                request_rank(
                    num_tokens_per_owner, position_ids[rank], x_hc[rank], input_ids[rank],
                    ori_block_table[rank], hca_cmp_block_table[rank], csa_cmp_block_table[rank], idx_block_table[rank],
                    hca_compress_state_block_table[rank], csa_compress_state_block_table[rank], csa_inner_compress_state_block_table[rank],
                    header_window, input_window, ids_window, tables_window, ready,
                    hidden_window, tail_window, complete, released,
                    hidden_out[rank], tail_out[rank], audit[rank],
                    request_owner, rank,
                    device=rank,
                )


def golden_requests(tensors):
    counts = tensors["num_tokens_per_owner"].tolist()
    owners = [rank for rank, count in enumerate(counts) if count > 0]
    tensors["hidden_out"].zero_()
    tensors["tail_out"].zero_()
    for owner in owners:
        length = counts[owner]
        tensors["hidden_out"][owner, :length].copy_(tensors["x_hc"][owner, :length, :D].to(torch.bfloat16))
        tail_length = min(TAIL_ROWS, length)
        tensors["tail_out"][owner, :tail_length].copy_(tensors["x_hc"][owner, length - tail_length : length])
    tensors["audit"].zero_()
    for owner in owners:
        length = counts[owner]
        span = max(TAIL_ROWS, (length + NUM_SEGMENTS - 1) // NUM_SEGMENTS)
        starts = [segment * span for segment in range(NUM_SEGMENTS)]
        lengths = [max(0, min(span, length - start)) for start in starts]
        for rank in range(CP_SIZE):
            out = tensors["audit"][rank, owner]
            out[1, :5] = torch.tensor([1, owner, length, span, tensors["position_ids"][owner, 0].item()])
            for part, segment in enumerate((rank, NUM_SEGMENTS - 1 - rank)):
                active = lengths[segment]
                start = starts[segment]
                dest = part * MAX_SEGMENT_TILES * TAIL_ROWS
                out[0, dest * 2 : (dest + active) * 2].copy_(
                    tensors["input_ids"][owner, 0, start : start + active].view(torch.int32)
                )
            for index, name in enumerate(TABLES, 2):
                value = tensors[name][owner, 0]
                out[index, : value.numel()].copy_(value)


def build_tensor_specs(counts, start_pos):
    positions = torch.arange(TOKENS, dtype=torch.int32).unsqueeze(0)
    positions = positions + torch.arange(CP_SIZE, dtype=torch.int32).unsqueeze(1) * 128 + start_pos
    ids = (2**40 + torch.arange(CP_SIZE * TOKENS, dtype=torch.int64)).reshape(CP_SIZE, 1, TOKENS)
    rows = torch.arange(CP_SIZE * TOKENS, dtype=torch.float32).reshape(CP_SIZE, TOKENS, 1) * 0.01
    columns = torch.arange(CP_REQUEST_HC_DIM, dtype=torch.float32).reshape(1, 1, -1) * 0.0001
    specs = [
        TensorSpec("num_tokens_per_owner", [CP_SIZE], torch.int32, init_value=counts),
        TensorSpec("position_ids", [CP_SIZE, TOKENS], torch.int32, init_value=positions),
        TensorSpec("x_hc", [CP_SIZE, TOKENS, CP_REQUEST_HC_DIM], torch.float32, init_value=rows + columns),
        TensorSpec("input_ids", [CP_SIZE, 1, TOKENS], torch.int64, init_value=ids),
    ]
    for name, size in zip(TABLES, TABLE_COLS):
        table = torch.stack([torch.arange(size, dtype=torch.int32).roll(rank + 3) for rank in range(CP_SIZE)])
        table[:, 1] = -1
        specs.append(TensorSpec(name, [CP_SIZE, 1, size], torch.int32, init_value=table.unsqueeze(1)))
    specs.extend(
        [
            TensorSpec("hidden_out", [CP_SIZE, TOKENS, D], torch.bfloat16),
            TensorSpec("tail_out", [CP_SIZE, TAIL_ROWS, CP_REQUEST_HC_DIM], torch.float32),
            TensorSpec("audit", [CP_SIZE, CP_SIZE, 9, CP_REQUEST_TABLE_COLS], torch.int32),
        ]
    )
    return specs


def compare_exact(actual, expected, **_kwargs):
    return torch.equal(actual, expected), "Exact request and cache-owner isolation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("-d", "--device", default=",".join(str(rank) for rank in range(CP_SIZE)))
    parser.add_argument("--cp", type=int, default=CP_SIZE)
    parser.add_argument("--ep", type=int, default=CP_SIZE)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != CP_SIZE or args.cp != CP_SIZE or args.ep != CP_SIZE:
        parser.error("The device count and CP/EP group sizes must agree.")
    torch.set_num_threads(4)
    cases = [
        (0, "none", 0),
        (129, "last", 0),
        (129, "all", 0),
        (50, "all", 1024),
        (257, "ends", 1152),
        (CP_REQUEST_CAPACITY, "first", 0),
        (178, "all", 1024),
    ]
    runtime_dir = None
    for length, owners, start_pos in cases:
        counts = torch.zeros(CP_SIZE, dtype=torch.int32)
        if owners == "none":
            pass
        elif owners == "all":
            counts[:] = torch.tensor([length + rank * 17 for rank in range(CP_SIZE)], dtype=torch.int32)
        elif owners == "last":
            counts[-1] = length
        elif owners == "first":
            counts[0] = length
        else:
            counts[0], counts[-1] = length, 50
        print(f"CP request case: counts={counts.tolist()}, base={start_pos}", flush=True)
        result = run(
            fn=request_host,
            specs=build_tensor_specs(counts, start_pos),
            golden_fn=golden_requests,
            compare_fn={name: compare_exact for name in ("hidden_out", "tail_out", "audit")},
            compile_only=args.compile_only,
            runtime_dir=runtime_dir,
            save_data=False,
            config=dict(platform=args.platform, distributed_config=DistributedConfig(device_ids=devices)),
        )
        if not result.passed:
            raise RuntimeError(result.error)
        if args.compile_only:
            return
        runtime_dir = str(result.work_dir)
    print("CP request isolation and repeated-window validation PASS", flush=True)


if __name__ == "__main__":
    main()

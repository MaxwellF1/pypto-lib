# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reset reserved decode padding pages and compressor state on device."""

import pypto.language as pl


@pl.jit.inline(auto_scope=False)
def reset_reserved_pages(
    cache: pl.Tensor,
    c2a_table: pl.Tensor,
    source_count: pl.constexpr,
    c2a_sources: pl.constexpr,
    page_width: pl.constexpr,
    dtype: pl.constexpr,
    pattern: pl.constexpr,
    tile_width: pl.constexpr,
    element_bytes: pl.constexpr,
    raw_dtype: pl.constexpr,
):
    rows = pl.tensor.dim(cache, 0)
    pages = rows // source_count
    flat = pl.reshape(cache, [rows, page_width])
    with pl.spmd(1, name_hint="reset_decode_padding"):
        worker = pl.tile.get_block_idx()
        words = pl.tile.full([1, tile_width * element_bytes // 4], dtype=pl.INT32, value=pattern)
        raw = pl.reinterpret_view(words, raw_dtype)
        fill = pl.reinterpret_view(raw, dtype)
        for source in pl.range(worker, source_count, 1):
            page = (source + 1) * pages - 1
            if source < c2a_sources:
                page = source * pages + pl.tensor.dim(c2a_table, 1) - 1
            for column in pl.range(page_width // tile_width):
                flat = pl.store(fill, [page, column * tile_width], flat)
    return cache


PAYLOAD_ROWS = pl.dynamic("PADDING_PAYLOAD_ROWS")
SCALE_ROWS = pl.dynamic("PADDING_SCALE_ROWS")
INDEX_ROWS = pl.dynamic("PADDING_INDEX_ROWS")
STATE_ROWS = pl.dynamic("PADDING_STATE_ROWS")
PAYLOAD_WIDTH = 65536
SCALE_WIDTH = 2048
INDEX_WIDTH = 512
STATE_WIDTH = 4096


@pl.jit(auto_scope=False)
def reset_padding_pages(
    payload: pl.InOut[pl.Tensor[[PAYLOAD_ROWS, PAYLOAD_WIDTH], pl.FP8E4M3FN]],
    scale: pl.InOut[pl.Tensor[[SCALE_ROWS, SCALE_WIDTH], pl.BF16]],
    index_scale: pl.InOut[pl.Tensor[[INDEX_ROWS, INDEX_WIDTH], pl.FP8E8M0]],
    state: pl.InOut[pl.Tensor[[STATE_ROWS, STATE_WIDTH], pl.FP32]],
    c2a_table: pl.Tensor[[1, 2], pl.INT32],
):
    payload_out = reset_reserved_pages(payload, c2a_table, 40, 0, PAYLOAD_WIDTH, pl.FP8E4M3FN, 0, 1024, 1, pl.INT8)
    scale_out = reset_reserved_pages(scale, c2a_table, 4, 3, SCALE_WIDTH, pl.BF16, 1065369472, 1024, 2, pl.INT8)
    index_out = reset_reserved_pages(index_scale, c2a_table, 8, 3, INDEX_WIDTH, pl.FP8E8M0, 2139062143, 512, 1, pl.UINT8)
    state_out = reset_reserved_pages(state, c2a_table, 3, 0, STATE_WIDTH, pl.FP32, 0, 1024, 4, pl.INT8)
    return payload_out, scale_out, index_out, state_out


def golden_reset_padding(tensors):
    import torch

    for name, sources in (("payload", 40), ("state", 3)):
        tensor = tensors[name]
        pages = tensor.shape[0] // sources
        tensor[pages - 1::pages].view(torch.uint8).zero_()
    scale = tensors["scale"]
    pages = scale.shape[0] // 4
    for source in range(4):
        page = source * pages + (1 if source < 3 else pages - 1)
        scale[page].fill_(1)
    index_scale = tensors["index_scale"]
    pages = index_scale.shape[0] // 8
    for source in range(8):
        page = source * pages + (1 if source < 3 else pages - 1)
        index_scale[page].view(torch.uint8).fill_(127)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    payload = torch.full((120, PAYLOAD_WIDTH), 19, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale = torch.full((12, SCALE_WIDTH), 19, dtype=torch.bfloat16)
    index_scale = torch.full((24, INDEX_WIDTH), 19, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    state = torch.full((9, STATE_WIDTH), 19, dtype=torch.float32)
    return [
        TensorSpec("payload", list(payload.shape), payload.dtype, init_value=payload),
        TensorSpec("scale", list(scale.shape), scale.dtype, init_value=scale),
        TensorSpec("index_scale", list(index_scale.shape), index_scale.dtype, init_value=index_scale),
        TensorSpec("state", list(state.shape), state.dtype, init_value=state),
        TensorSpec("c2a_table", [1, 2], torch.int32, init_value=torch.zeros((1, 2), dtype=torch.int32)),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate reserved decode padding page resets.")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    result = run(
        fn=reset_padding_pages,
        specs=build_tensor_specs(),
        golden_fn=golden_reset_padding,
        compile_only=args.compile_only,
        config=dict(backend_type=args.backend_type, device_id=args.device),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)

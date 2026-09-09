# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CP tail exchange stays correct across serving-style retained-window reuse."""

import os
from pathlib import Path
import subprocess
import sys


def _run_device_case(build_dir):
    import json

    HERE = build_dir
    LIB = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(LIB / "models/deepseek_v4_flash_mtp"), str(LIB)]
    sys.argv += ["--cp", "2", "--ep", "2", "--tp", "2"]
    import torch
    import pypto.language as pl
    import pypto.language.distributed as pld
    from pypto.runtime import RunConfig
    from pypto.ir import DistributedConfig
    from prefill_cp_exchange import _prefill_cp_hidden_tail_exchange_wave as exchange
    from prefill_cp_exchange import _clear_prefill_cp_exchange_signals as clear_signals
    from prefill_cp_exchange import D, CP_SIZE, TAIL_ROWS, LOCAL_PARTS, NUM_SEGMENTS, CP_TAIL_WINDOW_ROWS

    @pl.jit
    def rank_exchange(
        source: pl.Tensor[[LOCAL_PARTS * TAIL_ROWS, D], pl.BF16],
        reverse: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
        owners: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
        output: pl.Out[pl.Tensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16]],
        state: pl.Out[pl.Tensor[[2, 16], pl.INT32]],
        window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
        ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
        consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
    ):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="request_tail") as exchanged:
            for peer in pl.range(CP_SIZE):
                pl.write(state, [0, peer], pl.read(ready, [peer, 0]))
                pl.write(state, [1, peer], pl.read(consumed, [peer, 0]))
            output = exchange(source, reverse, owners, window, ready, consumed, output, rank, 0, 0)
        anchor = pl.create_tensor([1, 1, 8], dtype=pl.FP32)
        anchor_flat = pl.reshape(anchor, [1, 8])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="tail_completion", deps=[exchanged]):
            anchor_flat[0:1, 0:8] = pl.full([1, 8], dtype=pl.FP32, value=0.0)
        cleared = clear_signals(anchor, ready, consumed, pl.cast(1, pl.INT32), rank)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="after_clear", deps=[cleared]):
            for peer in pl.range(CP_SIZE):
                pl.write(state, [0, peer], pl.read(ready, [peer, 0]))
                pl.write(state, [1, peer], pl.read(consumed, [peer, 0]))
        return output

    @pl.jit.host
    def request(
        source: pl.Tensor[[CP_SIZE, LOCAL_PARTS * TAIL_ROWS, D], pl.BF16],
        reverse: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
        owners: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
        output: pl.Out[pl.Tensor[[CP_SIZE, CP_TAIL_WINDOW_ROWS, D], pl.BF16]],
        state: pl.Out[pl.Tensor[[CP_SIZE, 2, 16], pl.INT32]],
    ):
        hidden_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
        ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
        consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(CP_SIZE):
            window = pld.window(hidden_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
            ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
            consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
            rank_exchange(
                source[rank],
                reverse,
                owners,
                output[rank],
                state[rank],
                window,
                ready,
                consumed,
                rank,
                device=rank,
            )

    ids = [int(x) for x in os.environ["TASK_DEVICE"].split(",")]
    assert len(ids) == 2
    variant = "cleared"
    config = RunConfig(
        platform="a2a3",
        save_kernels=True,
        save_kernels_dir=str(HERE / ("build-" + variant)),
        distributed_config=DistributedConfig(device_ids=ids, num_sub_workers=0),
        ring_heap=(2**30,) * 4,
        ring_dep_pool=16384,
        ring_task_window=16384,
    )
    compiled = request.compile(config=config)
    source = torch.empty((CP_SIZE, LOCAL_PARTS * TAIL_ROWS, D), dtype=torch.bfloat16).share_memory_()
    reverse = torch.tensor([0, 2, 3, 1], dtype=torch.int32).share_memory_()
    owners = torch.tensor([0, 1, 1, 0], dtype=torch.int32).share_memory_()
    output = torch.empty((CP_SIZE, CP_TAIL_WINDOW_ROWS, D), dtype=torch.bfloat16).share_memory_()
    state = torch.zeros((CP_SIZE, 2, 16), dtype=torch.int32).share_memory_()
    rows = []
    with compiled.prepare(config=config, persistent=True, reset_persistent_windows=False) as worker:
        for repeat in range(3):
            for rank in range(CP_SIZE):
                for part in range(LOCAL_PARTS):
                    source[rank, part * TAIL_ROWS : (part + 1) * TAIL_ROWS].fill_(
                        repeat * 16 + rank * 2 + part + 1
                    )
            output.fill_(-1)
            state.zero_()
            worker(source, reverse, owners, output, state)
            expected = torch.cat(
                [source[0, :TAIL_ROWS], source[1, :TAIL_ROWS], source[1, TAIL_ROWS:], source[0, TAIL_ROWS:]]
            )
            row = dict(
                repeat=repeat,
                mismatch=int((output != expected).sum()),
                retired_signal_state=state[:, :, :CP_SIZE].tolist(),
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    (HERE / (variant + "-result.json")).write_text(json.dumps(rows, indent=2) + "\n")
    assert all(row["mismatch"] == 0 for row in rows), rows
    assert all(
        state == 0 for row in rows for rank in row["retired_signal_state"] for bank in rank for state in bank
    ), "Request retirement left stale credits"
    print("RETAINED THREE REQUESTS PASS", flush=True)


def test_prefill_cp_retained_windows(tmp_path):
    import pytest

    devices = [item for item in os.environ.get("TASK_DEVICE", "").split(",") if item]
    if len(devices) != 2:
        pytest.skip("Requires exactly two task-submit NPU devices and the a2a3 toolchain")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--device-case", str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RETAINED THREE REQUESTS PASS" in result.stdout


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--device-case":
        raise SystemExit("Run with pytest under task-submit --device <two devices>")
    _run_device_case(Path(sys.argv[2]))

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU checks of CP ownership, causal indices and final-window restoration."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    getattr(sys.modules.get("pypto"), "__pypto_stub__", False),
    reason="metadata builders import the real PyPTO kernel modules",
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "deepseek_v4_flash_mtp"


def _run(script):
    # Model imports freeze topology; isolate them from other contract modules.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(MODEL)]))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr




def test_cp_layout_exhaustive_lengths_and_invalid_inputs():
    _run('''
from prefill_cp_zigzag import cp_segment_layout, cp_owner_rank, cp_owner_part

for cp in (1, 2, 4, 8):
    for length in range(1, cp * 1024 + 1):
        span, starts, lengths = cp_segment_layout(length, cp)
        # Independently distribute a right-padded global token stream.
        padded = max(length, 2 * cp * 128)
        while padded % (2 * cp):
            padded += 1
        assert span * 2 * cp == padded
        assert 128 <= span <= 512
        assert sum(lengths) == length
        restored = []
        owners = set()
        for segment, (start, active) in enumerate(zip(starts, lengths)):
            rank = cp_owner_rank(segment, cp)
            part = cp_owner_part(segment, cp)
            owners.add((rank, part))
            assert segment == (rank if part == 0 else 2 * cp - 1 - rank)
            assert 0 <= active <= span
            restored.extend(range(start, start + active))
        assert len(owners) == 2 * cp
        assert restored == list(range(length)), (cp, length)
        last_segment = max(i for i, n in enumerate(lengths) if n)
        assert starts[last_segment] + lengths[last_segment] - 1 == length - 1

for cp in (1, 2, 4, 8):
    for length in (0, -1, cp * 1024 + 1):
        try:
            cp_segment_layout(length, cp)
        except ValueError:
            pass
        else:
            raise AssertionError((cp, length))
for length, cp in ((True, 8), (1.5, 8), ("1", 8), (1, True), (1, 8.0)):
    try:
        cp_segment_layout(length, cp)
    except TypeError:
        pass
    else:
        raise AssertionError((length, cp))
for cp in (0, 3, 16):
    try:
        cp_segment_layout(1, cp)
    except ValueError:
        pass
    else:
        raise AssertionError(cp)
''')


def test_local_prefill_keeps_world_rank_weights_and_request_metadata():
    _run('''
import inspect
import sys
sys.argv += ["--cp", "1", "--ep", "2", "--tp", "2", "--num-layers", "43"]
import prefill_fwd as fwd

assert (fwd.CP_FWD_CP_SIZE, fwd.N_RANKS, fwd.LM_HEAD_TP_SIZE) == (1, 2, 2)
parameters = inspect.signature(fwd.l3_prefill_fwd._func).parameters
assert (fwd.HCA_NUM_LAYERS, fwd.CSA_NUM_LAYERS) == (20, 21)
for name, page_rows in (("hca_cmp_kv", 1), ("csa_cmp_kv", 32)):
    shape = list(parameters[name].annotation.shape)
    assert shape[0] == 2 and shape[2:] == [page_rows, 1, 512]
    assert isinstance(shape[1], fwd.pl.Scalar), name
for name, suffix in {
    "kv_cache": [128, 1, 512],
    "hca_compress_state": [8, 1024],
    "csa_compress_state": [4, 2048],
    "csa_inner_compress_state": [4, 512],
    "idx_kv_cache": [32, 1, 128],
    "idx_kv_scale": [32, 1, 1],
}.items():
    shape = list(parameters[name].annotation.shape)
    assert shape[0] == 2 and shape[2:] == suffix, name
    assert isinstance(shape[1], fwd.pl.Scalar), name
for name in ("x_hc", "wq_a", "routed_w1", "routed_w3", "kv_cache", "hidden_out", "logits"):
    assert parameters[name].annotation.shape[0] == 2, name
for name in ("segment_starts_t", "owner_rank_table", "final_segment_t", "segment_active_lengths", "cache_owner_rank_t"):
    assert name not in parameters, name
assert list(parameters["num_tokens_per_owner"].annotation.shape) == [2]
''')
    _run('''
import sys
sys.argv += ["--cp", "2", "--ep", "4", "--tp", "2", "--num-layers", "4"]
try:
    import prefill_cp
except AssertionError as error:
    assert "CP=1 or CP=world" in str(error)
else:
    raise AssertionError("intermediate CP subgroups are outside the accepted contract")
''')


def test_swa_varlen_indices_restore_global_causal_windows():
    _run('''
import torch
import prefill_swa as swa

torch.set_num_threads(1)
for length in (1, 127, 128, 129, 2047, 2048, 2049, 4096, 8191, 8192):
    tensors, ctx = swa.build_metadata(8, num_tokens=length)
    assert sum(ctx["lengths"]) == length
    q = tensors["query_position_ids"].reshape(-1, 128)
    valid_q = tensors["query_token_to_request"].reshape(-1, 128) == 0
    assert torch.equal(q[valid_q].sort().values, torch.arange(length, dtype=q.dtype))
    overlays = tensors["overlay_position_ids"].reshape(-1, swa.OVERLAY_ROWS)
    indices = tensors["swa_indices"].reshape(-1, 128, 128)
    valid_k = indices >= 0
    local_indices = torch.where(valid_k, indices - swa.OVERLAY_BASE, 0).long()
    assert torch.all(local_indices >= 0)
    assert torch.all(local_indices < swa.OVERLAY_ROWS)
    actual = torch.gather(overlays[:, None, :].expand(-1, 128, -1), 2, local_indices)
    actual = torch.where(valid_k, actual, -1)
    expected = q[:, :, None] - 127 + torch.arange(128)
    expected = torch.where(valid_q[:, :, None] & (expected >= 0), expected, -1)
    assert torch.equal(actual, expected), length
    # Independently reconstruct the global final window from exchanged tails.
    for row, (seg, src) in enumerate(zip(
        tensors["final_win_seg_src"].tolist(), tensors["final_win_row_src"].tolist()
    )):
        position = length - 128 + row
        if position < 0:
            assert seg == src == -1
            assert tensors["final_slot_mapping"][row] == -1
        else:
            assert tensors["segment_tail_positions"][seg, src] == position
            assert tensors["final_slot_mapping"][row] == swa.ring_phys_row(position)
    assert tensors["query_position_ids"].shape == (8, 2, 4, 128)
''')


def test_hca_varlen_compressed_boundaries_and_final_state():
    _run('''
import torch
import prefill_hca as hca
import prefill_swa as swa

torch.set_num_threads(1)
for length in (1, 127, 128, 129, 2049, 8192):
    meta = hca.build_hca_metadata(8, num_tokens=length)
    raw = hca._build_raw_attention_metadata(8, num_tokens=length)
    swa_meta, ctx = swa.build_metadata(8, num_tokens=length)
    for name, value in raw.items():
        assert torch.equal(value, swa_meta[name]), (length, name)
    assert meta["segment_lengths"].tolist() == ctx["lengths"]
    assert meta["segment_starts"].tolist() == ctx["starts"]
    positions = meta["segment_cmp_positions"]
    slots = meta["segment_cmp_slots"]
    valid = positions >= 0
    expected = torch.arange(127, length, 128, dtype=positions.dtype) if length >= 128 else positions.new_empty(0)
    assert torch.equal(positions[valid].sort().values, expected), length
    assert torch.equal(slots[valid], (positions[valid] + 1) // 128 - 1)
    owner, part = meta["final_owner_rank"], meta["final_owner_part"]
    live = int(meta["snapshot_valid"][owner, part])
    assert live == min(128, length)
    assert torch.equal(meta["snapshot_positions"][owner, part, :live], torch.arange(length-live, length, dtype=torch.int32))
    for rank in range(8):
        for part in range(2):
            segment = int(meta["owner_segments"][rank, part])
            if ctx["lengths"][segment] == 0:
                assert meta["snapshot_valid"][rank, part] == 0
                assert torch.all(meta["snapshot_positions"][rank, part] == -1)
''')


def test_csa_varlen_seeds_candidates_and_final_state():
    _run('''
import torch
import prefill_csa as csa
import prefill_swa as swa

torch.set_num_threads(1)
for length in (1, 3, 4, 5, 129, 2049, 8192):
    meta, ctx = csa._build_metadata_tensors(8, num_tokens=length)
    raw, raw_ctx = csa._build_raw_attention_metadata(8, num_tokens=length)
    swa_meta, swa_ctx = swa.build_metadata(8, num_tokens=length)
    assert ctx["lengths"] == raw_ctx["lengths"] == swa_ctx["lengths"]
    assert ctx["starts"] == raw_ctx["starts"] == swa_ctx["starts"]
    names = dict(query_positions="query_position_ids", query_requests="query_token_to_request",
                 overlay_positions="overlay_position_ids", overlay_requests="overlay_token_to_request")
    for name in ("query_positions", "query_requests", "overlay_positions", "overlay_requests",
                 "overlay_active_lengths", "swa_indices", "final_win_seg_src", "final_win_row_src", "final_slot_mapping"):
        assert torch.equal(raw[name], swa_meta[names.get(name, name)]), (length, name)
    boundaries = meta["boundary_positions"]
    valid = boundaries >= 0
    expected = torch.arange(3, length, 4, dtype=boundaries.dtype) if length >= 4 else boundaries.new_empty(0)
    assert torch.equal(boundaries[valid].sort().values, expected)
    assert int(meta["candidate_history"]) == length // 4
    assert torch.equal(meta["main_logical_slots"][valid], (boundaries[valid] + 1) // 4 - 1)
    for segment, (start, active) in enumerate(zip(ctx["starts"], ctx["lengths"])):
        n = int(meta["seed_lengths"][segment])
        assert n == (start % 4 + 4 if segment > 0 and active > 0 else 0)
        assert torch.equal(meta["seed_positions"][segment, :n], torch.arange(start-n, start, dtype=torch.int32))
    positions = meta["final_snapshot_positions"]
    assert torch.equal(positions[positions >= 0], torch.arange(max(0, length-csa.STATE_LEN), length, dtype=torch.int32))
    assert torch.all(meta["final_main_state_mapping"][:, positions < 0] == -1)
    assert torch.all(meta["final_inner_state_mapping"][:, positions < 0] == -1)
''')

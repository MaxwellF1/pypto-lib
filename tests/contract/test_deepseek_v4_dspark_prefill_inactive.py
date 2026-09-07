# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contract checks for empty prefill groups and retained EP participation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "deepseek_v4_flash_dspark"


def _function(module, name):
    tree = ast.parse((MODEL_DIR / f"{module}.py").read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<prefill-contract>", "exec"), namespace)


def _call_guards(node, guards=()):
    if isinstance(node, ast.If):
        for child in node.body:
            yield from _call_guards(child, (*guards, ast.unparse(node.test)))
        for child in node.orelse:
            yield from _call_guards(child, (*guards, f"not ({ast.unparse(node.test)})"))
    else:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            yield node, guards
        for child in ast.iter_child_nodes(node):
            yield from _call_guards(child, guards)


def test_empty_groups_skip_group_work_but_keep_ep_calls():
    forward = _function("prefill_fwd", "prefill_fwd")
    group_count_writes = [
        node for node in ast.walk(forward)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "group_tokens" for target in node.targets
        )
    ]
    assert len(group_count_writes) == 1
    calls = list(_call_guards(forward))
    attention = [(call, guards) for call, guards in calls if call.func.id.startswith("prefill_attention_")]
    assert len(attention) == 5
    for _, guards in attention:
        assert "group_tokens > 0" in guards
    moe_calls = [(call, guards) for call, guards in calls if call.func.id == "prefill_moe"]
    assert len(moe_calls) == 5
    for call, guards in moe_calls:
        assert not any("group_tokens" in guard for guard in guards)
        assert ast.unparse(call.args[-1]) == "group_tokens"
    for call, guards in calls:
        if call.func.id in {"hc_head", "lm_head", "retire_o_proj_weight_signals", "_copy_target_hc_row"}:
            assert "group_tokens > 0" in guards

    moe = _function("moe", "prefill_moe")
    for call, guards in _call_guards(moe):
        if call.func.id in {"hc_pre", "hc_post", "prefill_cp_token_allgather_step"}:
            assert "group_tokens > 0" in guards
        if call.func.id == "_moe_tile":
            assert not any("group_tokens" in guard for guard in guards)


@pytest.mark.parametrize("local_rows", [1, 127, 128, 129, 4096])
@pytest.mark.parametrize("group_tokens", [0, 1, 7, 8192])
def test_empty_groups_keep_physical_waves_with_zero_routes(local_rows, group_tokens):
    moe = _function("moe", "prefill_moe")
    wave_loop = next(node for node in moe.body if isinstance(node, ast.For))
    wave_scope = next(node for node in wave_loop.body if isinstance(node, ast.With))
    wave_prefix = []
    for node in wave_scope.body:
        wave_prefix.append(node)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "wave_rows_i32":
            break
    namespace = {
        "pl": SimpleNamespace(min=min, cast=lambda value, dtype: int(value), INDEX=None, INT32=None),
        "T": 128, "local_rows": local_rows, "group_tokens": group_tokens,
    }
    wave_count = next(
        node for node in moe.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "num_waves"
    )
    _execute([wave_count], namespace)
    rows = []
    for wave in range(namespace["num_waves"]):
        namespace["wave_i32"] = wave
        _execute(wave_prefix, namespace)
        rows.append(namespace["wave_rows_i32"])
    assert len(rows) == (local_rows + 127) // 128
    assert sum(rows) == (local_rows if group_tokens else 0)
    assert all(0 <= count <= 128 for count in rows)


def test_empty_terminal_outputs_overwrite_stale_buffers_including_vocab_tail():
    forward = _function("prefill_fwd", "prefill_fwd")
    terminal = next(
        node for node in ast.walk(forward)
        if isinstance(node, ast.If) and any(
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "lm_head"
            for child in node.body for call in ast.walk(child)
        )
    )
    namespace = {
        "pl": SimpleNamespace(
            spmd=range, range=range, tensor=SimpleNamespace(dim=lambda tensor, axis: tensor.shape[axis]),
            full=lambda shape, dtype, value: torch.full(shape, value, dtype=dtype),
            BF16=torch.bfloat16, FP32=torch.float32, INT32=torch.int32,
        ),
        "D": 8, "MAIN_HIDDEN_DIM": 24, "local_tokens": 3, "MAX_LOGIT_ROWS": 2, "LM_HEAD_VOCAB": 129280,
        "LOGITS_ZERO_TILE": 4096, "SAMPLED_IDS_PAD": 16,
        "dspark_target_hidden": torch.full((3, 24), float("nan"), dtype=torch.bfloat16),
        "x_out": torch.full((3, 8), float("nan"), dtype=torch.bfloat16),
        "logits": torch.full((2, 129280), float("nan")),
        "sampled_ids": torch.full((2, 16), 123, dtype=torch.int32),
    }
    namespace["pl"].spmd = lambda count, **kwargs: range(count)
    _execute(terminal.orelse, namespace)
    assert torch.all(namespace["dspark_target_hidden"] == 0)
    assert torch.all(namespace["x_out"] == 0)
    assert torch.all(namespace["logits"] == 0)
    assert torch.all(namespace["sampled_ids"] == -1)


def test_inactive_state_oracle_rejects_finite_cache_writes():
    namespace = {}
    _execute([_function("prefill_fwd", "finite_tensor_compare")], namespace)
    compare = namespace["finite_tensor_compare"]
    expected = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    actual = expected.clone()
    actual[:2] += 1
    inputs = {"query_start_loc": torch.tensor([[0, 7], [0, 7], [0, 0], [0, 0]])}
    assert compare(actual, expected, preserve_inactive=True, inputs=inputs)[0]
    actual[2, 0] += 1
    assert not compare(actual, expected, preserve_inactive=True, inputs=inputs)[0]


def test_inactive_write_only_workspace_is_not_a_computed_result():
    namespace = {}
    _execute([_function("prefill_fwd", "finite_tensor_compare")], namespace)
    actual = torch.tensor([[1.0], [float("nan")]])
    inputs = {"query_start_loc": torch.tensor([[0, 7], [0, 0]])}
    assert namespace["finite_tensor_compare"](actual, torch.zeros_like(actual), inputs=inputs)[0]
    actual[0, 0] = float("nan")
    assert not namespace["finite_tensor_compare"](actual, torch.zeros_like(actual), inputs=inputs)[0]

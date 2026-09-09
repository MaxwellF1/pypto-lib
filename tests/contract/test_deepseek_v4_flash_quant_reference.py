# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU regression for the gate reference's symmetric INT8 tie rounding."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    getattr(sys.modules.get("pypto"), "__pypto_stub__", False),
    reason="gate imports the real PyPTO kernel modules",
)


@pytest.mark.parametrize("module, function", [
    ("gate", "_per_token_int8_quant"),
    ("utils", "int8_quant_per_row"),
])
def test_quant_half_amax_rounds_to_even(module, function):
    root = Path(__file__).resolve().parents[2]
    model = root / "models" / "deepseek_v4_flash_mtp"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), str(model)]))
    result = subprocess.run(
        [sys.executable, "-c", f'''
import torch
from {module} import {function} as quantize

# For a symmetric range [-127,127], half-amax maps to +/-63.5.
# Rounding to nearest even must produce +/-64. This BF16-representable
# maximum exposed scalar/amax lowering to reciprocal-times-scalar on CPU.
x = torch.tensor([[-1.703125, -.8515625, 0., .8515625, 1.703125]])
quantized, scale = quantize(x)
assert quantized.tolist() == [[-127, -64, 0, 64, 127]], quantized
assert torch.isfinite(scale).all() and (scale > 0).all()
'''],
        cwd=root, env=env, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run a V4.1 decode entry with eager or Graph HBG orchestration."""

import argparse
import ast
import hashlib
import json
import runpy
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "deepseek_v4_1_flash"
SCOPES = {
    "decode_c1a_full": ("c1a_full",),
    "decode_c1a_reindex": ("c1a_reindex",),
    "decode_c1a_reuse": ("c1a_reuse",),
    "decode_c2a_full": ("lib_decode_c2a_full",),
    "decode_c2a_reuse": ("lib_decode_c2a_reuse",),
    "decode_swa": ("lib_decode_swa",),
    "decode_moe": ("lib_decode_moe",),
    "decode_layer": ("lib_decode_layer_moe", "lib_decode_layer_swa"),
    "decode_fwd": (
        "v41_decode_encoder_00_04", "v41_decode_encoder_05_09",
        "v41_decode_encoder_10_14", "v41_decode_encoder_15_19",
        "v41_decode_decoder_20_24", "v41_decode_decoder_25_29",
        "v41_decode_decoder_30_34", "v41_decode_decoder_35_39",
        "v41_decode_final",
    ),
}


def is_graph_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pl"
        and node.func.attr == "graph"
    )


class UnwrapGraphScopes(ast.NodeTransformer):
    def __init__(self):
        self.calls = set()
        self.scopes = []

    def visit_With(self, node):
        contexts = [item for item in node.items if is_graph_call(item.context_expr)]
        if not contexts:
            return self.generic_visit(node)
        item = contexts[0]
        call = item.context_expr
        if (
            len(node.items) != 1 or item.optional_vars is not None or call.keywords
            or len(call.args) != 1 or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)
            or call.end_lineno != node.lineno or node.body[0].lineno <= node.lineno
        ):
            raise ValueError(f"Unsupported Graph scope at line {node.lineno}")
        self.calls.add(id(call))
        self.scopes.append((call.args[0].value, node))
        return node.body


def unwrap(source, path):
    original = ast.parse(source, filename=str(path))
    nodes = list(ast.walk(original))
    if any(isinstance(node, ast.Attribute) and ast.unparse(node) == "pl.jit.graph" for node in nodes):
        raise ValueError(f"Graph decorators are outside the eager source view contract in {path}")
    graph_calls = {id(node) for node in nodes if is_graph_call(node)}
    transformer = UnwrapGraphScopes()
    eager = transformer.visit(original)
    names = [name for name, _ in transformer.scopes]
    if graph_calls != transformer.calls or Counter(names) != Counter(SCOPES.get(path.stem, ())):
        raise ValueError(f"Unexpected Graph scope inventory in {path}: {names}")

    lines = source.splitlines(keepends=True)
    for _, scope in transformer.scopes:
        header = lines[scope.lineno - 1]
        _, separator, comment = header.partition("#")
        indent = " " * scope.col_offset
        lines[scope.lineno - 1] = indent + separator + comment if separator else "\n"
        body_indent = " " * scope.body[0].col_offset
        shift = len(body_indent) - len(indent)
        for index in range(scope.lineno, scope.end_lineno):
            if lines[index].startswith(body_indent):
                lines[index] = lines[index][shift:]
    result = "".join(lines)
    if ast.dump(ast.parse(result, filename=str(path))) != ast.dump(eager):
        raise ValueError(f"Eager source changes more than Graph scopes in {path}")
    return result, names


def prepare_eager_view():
    builds = ROOT / "build_output"
    builds.mkdir(exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix="a5hbg-eager-", dir=builds))
    manifest = {}
    for path in sorted(MODEL.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".cpp", ".cc", ".h", ".hpp"}:
            continue
        relative = path.relative_to(MODEL)
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source = path.read_bytes()
        data, scopes = source, []
        if path.suffix == ".py":
            eager, scopes = unwrap(source.decode("utf-8"), relative)
            data = eager.encode("utf-8")
        target.write_bytes(data)
        manifest[relative.as_posix()] = {
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "view_sha256": hashlib.sha256(data).hexdigest(),
            "removed_graph_scopes": scopes,
        }
    (view / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("eager", "graph"))
    parser.add_argument("entry", choices=sorted(SCOPES))
    parser.add_argument("model_args", nargs=argparse.REMAINDER, help="Arguments to the original entry")
    options = parser.parse_args()
    args = options.model_args
    if args and args[0] == "--":
        args = args[1:]
    if any(arg.startswith("--r") and "--runtime-dir".startswith(arg.split("=", 1)[0]) for arg in args):
        parser.error("This launcher compiles the selected mode. It does not accept --runtime-dir")
    if Path.cwd().resolve() != ROOT:
        parser.error("Run this command from the Lib checkout root")

    model = prepare_eager_view() if options.mode == "eager" else MODEL
    entry = model / (options.entry + ".py")
    print(f"A5 HBG {options.mode} source: {entry}", flush=True)
    sys.path[:0] = [str(model), str(ROOT)]
    sys.argv = [str(entry), *args]

    from pypto import passes

    with passes.PassContext([], runtime=passes.RuntimeKind.HOST_BUILD_GRAPH):
        runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()

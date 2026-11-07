# DeepSeek V4.1 Flash on A5 HBG

This branch keeps one model source tree for eager and Graph HBG. Both modes use
Simpler's `host_build_graph` runtime. The Graph mode compiles the model directly.
The eager mode builds an ignored source view under `build_output/a5hbg-eager-*`.
It removes the 18 named Graph scope wrappers and checks that the rest of each
Python AST is unchanged. The launcher reports the source path and records
source hashes in that view. Both modes use the same fixed Gate launch geometry.

The launcher compiles the selected mode and rejects `--runtime-dir`. Use the
original model entry for a separate replay and verify the saved mode and source
manifest first. The view retains source line numbers for debugging. Keep it while replaying a
build that refers to its C++ sources. Once the run and any replay are finished,
the generated view and the associated build can be removed together.

## Environment

Use the exact revisions in [versions.json](versions.json) and follow the
[installation guide](../get-started/installation.md) in an isolated environment.
Fetch the combination branch and select the validated commit explicitly:

```bash
git clone --branch a5hbg https://github.com/MaxwellF1/pypto.git pypto-a5hbg
git -C pypto-a5hbg checkout --detach c518868152128723dda6c272fecc061fb85f191b
git -C pypto-a5hbg submodule update --init --recursive
```

Install Simpler from the pinned PyPTO `runtime/` gitlink. Use the PTOAS version
and PTO ISA pin from the same checkout. Run from the Lib checkout root. The
launcher imports the installed PyPTO and forwards options to the original
model CLI. There is no separate model checkout for eager.

The [adaptation and PR tracking table](TRACKING.md) identifies which patches
are needed for this decode path and which evidence belongs to previous pins.

## Functional checks

Use your site's allocator. For example, save the following as `run-c1a.sh`
with the absolute paths for your site. The job itself activates the environment
and selects the checkout. The command uses the B12 × 8K C1A reuse case. Change
`graph` to `eager` for the ordinary HBG comparison.

```bash
#!/usr/bin/env bash
set -euo pipefail
source /path/to/site-a5-toolchain-env.sh
source /path/to/venv/bin/activate
cd /path/to/pypto-lib
python tools/a5hbg/run_model.py graph decode_c1a_reuse -- -p a5 -d "$TASK_DEVICE" \
  --start-pos 8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192,8192
```

On a shared server using `task-submit`, inspect the running tasks before
submitting. Set `idle_card` to an available healthy card from the fresh list
and submit using its explicit ID. When none is available, queue with auto
allocation instead.

```bash
task-submit --list
# Set idle_card from the fresh list before using this form.
task_id=$(task-submit --device "$idle_card" --max-time 900 "bash \"$PWD/run-c1a.sh\"")
task-submit --status "$task_id"

# Alternative when there is no available card.
task_id=$(task-submit --device auto --device-num 1 --max-time 900 "bash \"$PWD/run-c1a.sh\"")
task-submit --status "$task_id"
```

Check the generated orchestration C++ for `rt_submit_graph` call sites in the
Graph build and their absence in the eager build. Also check each physical
run's Native scheduler publication while its worker is open. Read
`scheduler_mode_publish_count` before and after the dispatch and
`scheduler_runtime_mode` after it. The qualified cases observed fresh mode `1`
(AICore), matching completed Native runs and Graph definitions in the Graph
arm. The runtime name and generated C++ identify the intended mode. The device
publication and completed run establish the observed path.

For MoE, request two cards and pass this model command to the allocator. The
arguments are the A5 Lib CI case, with 24 total experts and 72 logical token
rows per rank. The model supplies its original Torch reference and comparators.

```bash
python tools/a5hbg/run_model.py graph decode_moe -- -p a5 -d "$TASK_DEVICE" \
  --ep 2 --num-tokens 72 --experts-per-rank 12 --balanced-routing
```

Both checks must report their output comparisons and `[RUN] PASS`. For an
annotation/code-generation check, invoke the launcher directly with
`--compile-only` and a device argument of the expected cardinality. That path
compiles without launching a worker or reserving an accelerator. For the 40-layer synthetic execution smoke,
request two cards and use the original forward entry:

```bash
python tools/a5hbg/run_model.py graph decode_fwd -- -p a5 -d "$TASK_DEVICE" \
  --tp 1 --dp 2 --ep 2 --experts-per-rank 64 --weight-bank-size 40
```

The forward smoke checks execution and head outputs. Its pass line explicitly
states that full numerical accuracy was not validated. The branch carries the
matching prefill and DSpark callers to keep the model source snapshot coherent.
Their behavior is outside the A5 decode qualification scope.

## Verified configuration

The pinned combination passed the original CI C1A reuse B12 × 8K case on one
A5 card and the EP2 MoE case on two A5 cards in both modes. Every run published
resident AICore mode `1` and completed. Graph recorded one definition per worker
and eager recorded none. The receipt and source pins are in [versions.json](versions.json).

## Performance

Run the same case and mode pair on the same allocated card set. Use the existing
Lib benchmark path with `PYPTO_BENCH=1`, `PYPTO_BENCH_ROUNDS=100`, and
`PYPTO_BENCH_WARMUP=10`. Compare the same device-wall counter from both runs.
Collect traces in a separate dispatch. Keep mode-specific builds and replay
inputs with their own source hashes. Earlier measurements with the dynamic
eager Gate geometry are historical and do not describe this fixed-geometry
profile.

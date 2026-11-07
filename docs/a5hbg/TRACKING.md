# A5 HBG delivery and dependency tracking

Last reviewed: 2026-11-07 (Asia/Shanghai).

This tracks V4.1 Flash decode adaptations, their framework dependencies and measured acceptance. Use [the run guide](README.md) and `versions.json` for the runnable combination. Published PRs, merged support and model acceptance are separate states.

## Acceptance scope

The frozen A5 baseline passed original CI numerical comparisons for six attention operators and EP2 MoE in HBG eager and HBG Graph. The original SWA-plus-MoE `decode_layer` CI case passed all 16 comparators in each mode. Both modes executed all 40 `decode_fwd` layers with AICore scheduling.

The layer CI uses conditioned-stage oracles. Its 16 passing checks are stage-level acceptance, not an independent composed-layer Golden. The receipt also records an informational composed `x_next` comparison: relative L2 `0.14117299`, maximum absolute error `3.2990062`, identical in eager and Graph. That diagnostic is not an original CI acceptance gate. No numerical threshold was relaxed.

The 40-layer CLI uses 128 synthetic experts and checks execution/output padding. It has no whole-model Golden and does not establish trained-weight model accuracy. Prefill and serving are outside this acceptance. Unified-branch verification is recorded separately; an earlier result does not qualify a rebased PR or newly unified runner.

## Lib adaptations

All model paths below are relative to `models/deepseek_v4_1_flash/`. The source inventory distinguishes carried adaptations from already-upstream BF16 implementation files.

| Adaptation | Main source area | Purpose and boundary |
| --- | --- | --- |
| Graph regions | Six attention entry files, `decode_moe.py`, `decode_layer.py`, `decode_fwd.py` | 18 explicit scopes across nine files; eager is a mechanical scope-only derivative of the same maintained source. |
| Device-dependent workload sizes | `decode_moe.py`, `expert_routed.py` | Create a fixed upper-bound task population; read actual counts and skip invalid work inside kernels. Preserve valid rows and zero-fill tails. |
| Wait compatibility | `decode_moe.py`, `expert_routed.py`, `engram.py` | Three wait sites switch from `pld.system.defer_wait` to blocking `pld.system.wait` relative to the base. Deferred completion is not qualified here. |
| Communication ordering and routed workspaces | `decode_moe.py`, `expert_routed.py` | Order signal publishers before blocking waits and consumers after wait completion. Hoist receive/routed scratch to explicit caller-owned workspaces; retain stage completion dependencies across gather, W13, activation quantization, W2 and scatter. |
| Reindex routing | `decode_indexer.py`, `kernels/index_score_topk.cpp` | Compute route/workload metadata on device. Keep a recordable launch structure and let the dense/sparse kernel paths skip inactive work using that route. Host orchestration no longer reads the device-produced route. |
| Physical pool capacity | C1 full/reindex, indexer, layer and forward callers | Pass the actual pool block-id dimension instead of deriving capacity from logical active rows or an unrelated formula. |
| Allocation and projection extents | Attention helpers and `o_proj.py` | Carry the actual output-group capacity used by the slice into projection; retain generic dynamic-row behavior. |
| Gate precision and launch shape | `gate.py`, MoE/layer/forward weight specifications | Adopt the upstream BF16 gate/normalization interface and compatible weight banks. The shared profile uses fixed Gate workers/route tiles with invalid tail work skipped; old dynamic-eager timing is a different profile. |
| Layer CI interface | `decode_layer.py` | Keep Host-control scales on Host and map cache comparisons to physical capacity 80. The original layer0 CI has SWA and MoE Graph regions; other standalone layer branches are not claimed as separately accepted Graph cases. |
| 40-layer Graph population | `decode_fwd.py` | Split ten-layer chunks into five-layer regions to remain within the existing 1024-task bound: eight regions plus final head. Keep all 40 layers and weight-bank ordering. |
| 40-layer input fixture | `decode_fwd.py` | Build valid metadata for 72 rows before padding to 80; pad token/position tails with zero and inactive cache/slot tails with -1. The CLI remains synthetic execution validation. |
| Packed scratch geometry | `decode_fwd.py` | Use 1D allocations/slices before reshape instead of a singleton-leading 2D view. Element counts, byte offsets and reshape targets remain unchanged; this avoids the existing singleton-stride defect. |


### Model source provenance

The development branch preserves one coherent 88-file V4.1 model family. Its repository base is `d9a04f8b1d4cb62220e0f9b0b31d971dad9b0746`. The historical donor starts from Lib `740bc389b753ff9c0522c479390a6e00aa52c52f`, the gitlink in [Serving PR #305](https://github.com/hw-native-sys/pypto-serving/pull/305) at historical head `29fed17d99cb4b97c9be3127c149e630aa158f35`, and includes the upstream BF16 closure and later A5 decode work. The branch does not adopt that Serving change.

The raw donor differs from the repository base in 50 model files. Four differences only change CPU-simulation code; this branch keeps the base versions of `hc_collapse_norm.cpp`, `idx_k_norm_rope_publish.cpp`, `o_proj_a.cpp` and `o_quant_vf.cpp`. Their device-code text is equivalent. The resulting 46 model differences comprise 29 decode-closure files and 17 inherited prefill/DSpark compatibility files. The latter preserve a coherent set of shared helper signatures and have no new acceptance here.

The branch has four native source differences relative to the base. Eager and Graph share these exact implementations. The main A5 change is in Python DSL orchestration and buffer/workload descriptions.

| Native file | Provenance and purpose |
| --- | --- |
| `expert_routed_mxfp4.hpp` | Byte-identical to the historical donor; active expert packing dependency. |
| `q_lora_projection.hpp` | Byte-identical to the historical donor; active projection/pipe dependency. |
| `index_score_topk.cpp` | Earlier A5 adaptation: bound active shards by allocated capacity and pass the device route/mode into the kernel. Matching DSL argument slots are required. Score and Top-K math are retained. |
| `q_norm_quant_decode.cpp` | Already present in the earlier A5 donor. Reuse the normalization/quantization kernel; the entry clamps live rows minus tile offset to `[0, 16]` for fixed task launch and padded tails. |

The shared Golden harness and repository tests are unchanged. Decode acceptance retains the frozen test method, including its two stricter Layer `max_error_ratio` values (`0.020`/`0.015`, versus base `0.025`/`0.020`).

See [the PR identity snapshot](PR_STATUS.json) for exact head and merge identities. Table hashes are PR heads, not merge commits.

## PyPTO PR map

Required means the equivalent capability is needed by the indicated path. The pinned integration ref selects the actual combination; not every open PR below is a prerequisite for the current CLI.

| PR and snapshot state | Functional purpose | Resulting behavior | Current role |
| --- | --- | --- | --- |
| [#3323](https://github.com/hw-native-sys/pypto/pull/3323) — OPEN at `e29ae8bb` | Graph cache read/write dependencies | Propagate backing-storage reads and writes through views, calls and loop carries. | Required equivalent fix for Graph cache correctness; reuse existing PR. |
| [#3322](https://github.com/hw-native-sys/pypto/pull/3322) — OPEN at `ffe7c0c2` | Distributed Graph communication | Preserve CommCtx boundary representation and its replay-linked scalar origin. | Required equivalent fix for multi-card Graph MoE; reuse existing PR. |
| [#3321](https://github.com/hw-native-sys/pypto/pull/3321) — OPEN at `6ce74c84` | JIT Submit and Graph identities | Specialize submitted callees and retain tensor metadata, constexpr and unique Graph symbols. | For these JIT Submit/SPMD Submit forms; the currently qualified decode source bundle did not include this PR. |
| [#3277](https://github.com/hw-native-sys/pypto/pull/3277) — OPEN at `fc39d42c` | Explicit Graph predecessors and completion | Support graph deps/as-done and Graph Submit ordering without changing recorded identity. | For explicit invocation TaskIds; this explicit invocation interface is outside current model CI. |
| [#3276](https://github.com/hw-native-sys/pypto/pull/3276) — OPEN at `fd4e3d46` | HOST to CHIP result binding | Bind results to actual returned callee parameters instead of writable-argument order. | For chained HOST/CHIP Call/Submit return forms. |
| [#3275](https://github.com/hw-native-sys/pypto/pull/3275) — MERGED at `970fa6b1` | Graph GM-pipe capacity | Size hoisted pipe allocations from concrete launch consumers and core_num. | Merged; retain this behavior in new compiler patches. |
| [#3190](https://github.com/hw-native-sys/pypto/pull/3190) — MERGED at `a4a4f633` | Existing HOST control tensors | Accept Simpler HOST descriptors with NONE transfer, rejecting device control handles. | Merged; does not support mixed Host/device use of one parameter. |
| [#3189](https://github.com/hw-native-sys/pypto/pull/3189) — MERGED at `c132be69` | Existing write-only Graph output storage | Accept supplied Out buffers without mistaking them for runtime allocation. | Merged; Graph output binding foundation. |
| [#3188](https://github.com/hw-native-sys/pypto/pull/3188) — MERGED at `1a067a82` | Nested Host-control yields | Resolve For/If/While yields through lexical parents and preserve carried aliases. | Merged; nested Host-control analysis. |
| [#2369](https://github.com/hw-native-sys/pypto/pull/2369) — MERGED at `165a8a7d` | Deferred-completion wait DSL | Register dedicated AIV waiters through AsyncCtx; continuation uses TaskId dependencies. | Merged; this A5 Lib profile uses blocking wait instead. |
| [#2368](https://github.com/hw-native-sys/pypto/pull/2368) — CLOSED at `f0b40e95` | Earlier deferred-wait proposal | Closed proposal for the deferred-completion interface. | Use merged delivery #2369 rather than this branch. |

The new compiler PR is prepared separately for review of fixed Graph metadata, terminal Graph output liveness and nested `tensor.dim` code generation with allocation ordering. Publication requires its own successful build and regression checks.

The standalone compiler commit `44b61bf52d5a2cb6b4809cc6a8d8c9b8f5757ca6` is included in the delivery combination. It passed 68 focused tests. Its upstream PR has a separate publication check.

### Follow-up on existing PyPTO PRs

- [#3322 review job](https://github.com/hw-native-sys/pypto/actions/runs/422234124899/job/127947327733) did not complete the automated review. Source review confirms a separate Graph-body `get_comm_ctx` alias gap: materialization can leave a CommCtx alias that the scalar-only rewrite does not fold. The dedicated regression fails at the unchanged PR head, then passes with the candidate alias fix; 166 focused tests passed, including generated C++ syntax against the pinned runtime headers. The candidate is kept separate from the decode integration ref. The qualified V4.1 cases use automatic context companions and do not exercise the explicit-query case.
- [#3323 model CI](https://github.com/hw-native-sys/pypto/actions/runs/422256798187/job/128027859383) reports four numerical mismatches; the PR discussion attributes them to CSA prefill workspaces. Its [A5 onboard check](https://github.com/hw-native-sys/pypto/actions/runs/422256798187/job/128027859287) fails at artifact finalization with `ECONNRESET`. Source review also confirms two separate tuple-lineage gaps: single-tuple returns can lose per-element storage roots, and a direct tuple projection in a loop yield can lose carry effects. They belong in this existing PR. Current decode integration needs its own boundary-direction checks; the findings alone do not establish a decode regression. Do not mark all CI green.
- The source/CI states above are snapshots. `PR_STATUS.json` records which head each review/check snapshot covers.

## Simpler PR map

The frozen model qualification uses clean runtime `391b2fbcd4dd053204a677662c39b6b7636e8043`. Validation published on open PRs is separate and is not a model run of those heads.

| PR and snapshot state | Functional purpose | Resulting behavior | Current role |
| --- | --- | --- | --- |
| [#2934](https://github.com/hw-native-sys/simpler/pull/2934) — OPEN at `a8198c5d` | Async communication-buffer lifetime | Retain native-domain ownership until registered consumers complete, and quarantine uncertain cleanup. | Async multi-run preparation; not included in the frozen synchronous 40-layer result. |
| [#2932](https://github.com/hw-native-sys/simpler/pull/2932) — OPEN at `1a908254` | Large configured task windows | Reserve ready queues from configured capacity rather than retaining a hard 32768 cap. | For expanded task windows; current split model does not require it. |
| [#2931](https://github.com/hw-native-sys/simpler/pull/2931) — OPEN at `e1c801ae` | Host hazard-map growth | Grow ordinary/recorded hazard pools without moving producer entries or invalidating dependency links. | For larger live-buffer populations; no frozen-model qualification of this unmerged head. |
| [#2930](https://github.com/hw-native-sys/simpler/pull/2930) — OPEN at `6ae66d52` | Graph definition admission accounting | Count distinct published/inflight keys once to prevent premature 16-definition exhaustion. | For many reusable definitions and async admission; reuse existing PR. |
| [#2869](https://github.com/hw-native-sys/simpler/pull/2869) — MERGED at `9643297d` | Runtime explicit Graph predecessors | Attach invocation-specific predecessors to outer Graph completion and reject self edges. | Merged counterpart of PyPTO #3277. |
| [#2828](https://github.com/hw-native-sys/simpler/pull/2828) — MERGED at `a206b57e` | A2/A3 HBG SDMA prefetch | Expose existing DMA workspace addresses to HBG kernels through the dispatch context. | Merged A2/A3 V4 prefetch support; not a new A5 scheduler fix. |
| [#2743](https://github.com/hw-native-sys/simpler/pull/2743) — MERGED at `1153b449` | A3 attention Softmax correctness | Order vector reads before MTE2 overwrites reused score buffers in HBG and TMR kernels. | Merged A3 numerical fix; distinct from A5 BF16 gate adaptation. |
| [#2742](https://github.com/hw-native-sys/simpler/pull/2742) — MERGED at `62c70eca` | A3 queue-log overhead | Move routine QPROBE diagnostics to DEBUG while retaining warnings and enabled profiling. | Merged observability change; does not select an A5 scheduling backend. |
| [#2736](https://github.com/hw-native-sys/simpler/pull/2736) — MERGED at `777c4b6b` | Async activation/close race | Use compare-exchange so submission close cannot overwrite a newer EXECUTING phase. | Merged asynchronous completion and forward-progress fix. |

### Existing third-party fix

[Simpler #2146](https://github.com/hw-native-sys/simpler/pull/2146) addresses singleton-axis stride/contiguity across HBG, TMR and Graph descriptors. It is open and conflicting at the snapshot. Our isolated HBG draft has the same production condition and nine passing Host tests; no duplicate PR is needed. The qualified Lib uses one-dimensional packed scratch to avoid that dependency.

A5 resident AICore Graph expansion, AsyncCtx APIs and the 1088-byte CommContext layout already exist in the frozen runtime. Their presence does not qualify compiler-generated defer_wait on this model. Blocking wait and in-kernel routing remain deliberate Lib adaptations; mixed or multi-block dispatch predicates remain restricted.

## Versions and evidence

The historical baseline uses Core binary SHA256 `a3725667f9f69bd2aab12fd2bcae879026b5d8b8712b5e6dfa9bf986912736b4`, clean Simpler `391b2fb`, PTOAS v0.72 and PTO-ISA `f83d4b7e4be0a4ca9805944ae2d9ce8c84e581f7`. That Core was a source-patched workspace, not a public clean commit. `versions.json` records public refs for the unified development branch. Mutable branch names do not establish tested identities.

Keep tensors, complete traces, binaries and raw evidence remotely. Set `A5HBG_EVIDENCE_ROOT` to the existing `evidence/a5-graph-resident-v1` directory on the experiment machine. Verified indexes relative to that root:

| Check | Evidence index |
| --- | --- |
| Six attention operators, eager/Graph, original 8K CI | `ci-attention-canonical-a372-v1/receipt.json` |
| EP2 BF16 MoE, eager/Graph, original CI | `moe-ci-bf16-acceptance-a372-v3/qualified-summary.json` |
| Original layer CI, both modes, 384 total experts | `layer-graph-ci-v5/FINAL-RECEIPT.json` |
| 40-layer Graph synthetic CLI | `full40-ci-smoke-a372-v5/runs/numeric-graph/receipt.json` |
| 40-layer eager synthetic CLI | `full40-ci-smoke-a372-v6/runs/numeric-eager/receipt.json` |
| Independent completion/scheduler/source audit | `full40-ci-smoke-independent-closeout-v1.json` |
| Existing runtime support and singleton PR audit | `native-singleton-contiguity-draft-v1/upstream-audit-v1.md` |

The 40-layer Graph compilation has nine submit points: eight five-layer regions and the final head. Completion and scheduling mode were checked. Default device logs lacked raw GraphDefine records; these receipts do not provide an independent resident-DAG or cached-replay trace proof.

## Unified branch validation

Source derivation has been checked: exactly 18 intended scopes are removed for eager. The nine affected module ASTs match the frozen eager donor after scope removal; the fixed Gate launch is intentionally shared by both modes. On the historical a372 Core, scheduler task `task_20261107_143927_20848024664` completed with exit 0. The four original CLI compile-only arms (C1A reuse and EP2 MoE, eager and Graph) all use `host_build_graph`. Both Graph arms generate one `rt_submit_graph` site; both eager arms generate zero. There were no `InParamWritten` diagnostics, and all 4,246 source/binary guard entries matched before and after. See `a5hbg-lib-integration-v1/runs/cpu-v1/receipt.json` under the evidence root.

The delivery combination uses PyPTO `c518868152128723dda6c272fecc061fb85f191b` and Simpler `391b2fbcd4dd053204a677662c39b6b7636e8043`. The original C1A reuse and Layer0 CLIs compiled on this combination, with one and two Graph submissions respectively and zero `InParamWritten` diagnostics. See `core-a5hbg-integration-v1/cpu-receipt-v2.json`.

On A5, C1A reuse passed all six original CI Golden comparisons in each mode. EP2 MoE passed all three in each mode. All four runs recorded AICore mode 1 and native completion. Graph emitted one submission site per case and recorded Graph definitions on each participating card. Eager had no Graph sites or definitions. The 5,032 guard entries matched before and after. See `a5hbg-lib-integration-v1/runs/public-combo-numeric-v1/receipt.json` under the evidence root.

The published combination (PyPTO `c5188681`, Simpler `391b2fb`, Lib `aef9011`) also completed the original 40-layer `decode_fwd` synthetic fixture in eager and Graph on the same two A5 cards. Both ranks recorded AICore mode 1 and native completion in each run, and the CLI output checks passed. Graph generated nine submission sites and eager generated zero. The 128-expert fixture checks execution and output padding. It has no trained-weight whole-model Golden. See `core-a5hbg-integration-v1/full40-combo-c518868-v1/runs/graph/receipt.json` and the corresponding `runs/eager/receipt.json`.

## Performance evaluation

The functionality runs above are not benchmark results. The earlier C1 reuse AICore Graph report is retained remotely under `evidence/lib-kernel-acceptance-v1/reuse-perf-v1/report/REPORT.md`, relative to the task workspace. Its historical clean device-wall median is 943.328 microseconds. Historical TMR/eager figures use different pins/cards and are not a strict ranking. Legacy AICPU Graph results must not be relabeled AICore.

For new comparisons, hold Lib/dependency revisions, inputs, precision, device and warmup/round counts constant. Record device wall separately from the first-to-last AICore kernel span and Host wall. Effective-time extractors returning zero without required orchestration/scheduler windows produce N/A, not a zero-latency winner.

## Review priorities

| Capability | Required deliveries | Acceptance boundary |
| --- | --- | --- |
| Current A5 Graph decode | Existing PyPTO #3322/#3323 and the new metadata/liveness/codegen closure; merged output/pipe support is retained | Original operator/layer numerical checks and synthetic 40-layer execution; public combination has a separate validation record |
| Chained HOST/JIT and explicit Graph ordering | PyPTO #3276/#3321/#3277 as the selected caller uses those interfaces; Simpler #2869 is merged | These interfaces are not automatically qualified by the current model smoke |
| Async multi-run and larger capacity | Existing Simpler #2934 and #2930/#2931/#2932 for their specific lifecycle/capacity triggers | Requires dedicated multi-request/capacity acceptance, not just one synchronous decode |
| Whole-model and service delivery | Trained-weight whole-model Golden, prefill and serving integration | Still outstanding; do not infer it from padded-output or per-stage comparisons |

## Maintenance and next decisions

1. Keep a logical fix in its existing PR. Inspect its latest remote source/reviews before updating it, then rerun relevant checks.
2. Put uncovered metadata/liveness/codegen fixes in the new compiler PR. Retain upstream GM-pipe behavior and avoid duplicate CommCtx/backing-storage implementations.
3. Publish runnable fork combination refs separately from upstream review PRs. Update `versions.json`, this table and acceptance indexes together when the combination changes.
4. Refresh GitHub state, review and CI before merge decisions. Snapshot states are not continuous monitoring. Decode-only acceptance does not close the broader [four-repository track](https://github.com/hw-native-sys/simpler/issues/2658).
5. Remaining acceptance includes trained-weight 40-layer numerical accuracy, prefill and serving; async multi-request lifecycle and performance need their own qualification.

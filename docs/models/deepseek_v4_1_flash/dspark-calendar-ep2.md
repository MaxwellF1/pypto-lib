# DSpark decode on A5 with the calendar scheduler

The balanced EP2 DSpark forward reached **16.007 ms** median fastest-rank AICore wall with this Lib source snapshot and the frozen replay package described below. The Lib parent is `e6f15d56d5b613ea62abc983740ee721c4d3cc93`. The calendar runtime is [Simpler e06a25c](https://github.com/hw-native-sys/simpler/commit/e06a25c3c548d30366c0f44df5937213d51bb0e6). The optimizations change the work and data dependencies that the Lib submits. Physical lane placement remains a runtime decision.

## What changed along the forward path

| Stage | Implementation and effect |
| --- | --- |
| Router Gate | Both the token-bias and ordinary Gate launches drop the full-width `sync_start` requirement. Independent blocks can enter as their inputs and lanes become ready. The score and routing operations still run. [Gate](../../../models/deepseek_v4_1_flash/gate.py) |
| HC to QKV | `hc_qkv_quant` combines hyper-connection collapse, normalization, and quantization into one submitted stage. SWA, C1A reuse and C2A reuse emit the projection payload and scales directly. C1A reindex also emits the normalized BF16 tensor needed by its indexer. C1A full and C2A full retain their original front end. [HC](../../../models/deepseek_v4_1_flash/hc.py) · [QKV](../../../models/deepseek_v4_1_flash/qkv_proj_rope.py) · [Reindex caller](../../../models/deepseek_v4_1_flash/decode_c1a_reindex.py) |
| Sparse attention to output projection | SWA, C1A full/reindex/reuse and C2A full/reuse use `gather_project_hc_post`. The Gather epilogue applies the inverse rotary transform where needed and produces MXFP8 payload and scales in the format O-A consumes. This removes the full BF16 attention-output handoff and the separate O-A input preparation in these paths. It preserves the O-A quantization bound. O-A and O-B still perform their own matrix multiplications. [Attention](../../../models/deepseek_v4_1_flash/sparse_attention.py) · [Gather epilogue](../../../models/deepseek_v4_1_flash/kernels/sparse_quant_epilogue.hpp) · [Output projection](../../../models/deepseek_v4_1_flash/o_proj.py) |
| C1A full index selection | After scoring, a single `_deepseek_select_with_pool` submission produces both the selected attention indices and the candidate-pool IDs and counts used by later reindex layers. This replaces separate attention and pool selection launches in the candidate-producing path. [Indexer orchestration](../../../models/deepseek_v4_1_flash/decode_indexer.py) · [Native selector](../../../models/deepseek_v4_1_flash/kernels/lightning_indexer.cpp) |
| MegaMoE to HC | The fused expert path applies HC post-mixing after combine and writes the next four-stream hidden state inside the native invocation. The path still exposes the combined output. This removes a separate HC post submission from the fused path. [MoE caller](../../../models/deepseek_v4_1_flash/decode_moe.py) · [MegaMoE specialization](../../../models/deepseek_v4_1_flash/mega_moe.py) · [Native epilogue](../../../models/deepseek_v4_1_flash/kernels/moe_fused.cpp) |

## What the timings establish

Both comparisons use the complete `decode_fwd_dspark` forward on A5 with balanced routing, EP2/TP2/DP2, B12 per rank, 8K context, 72 target rows per rank, 12 target experts and 4 draft experts per rank. The metric is the median AICore wall on the fastest rank. Each row is a separate controlled comparison.

| Variable changed | Control | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Lib package, fixed dynamic Simpler #3234 at `118bc1b` | 18.691 ms | 17.379 ms | 1.312 ms (7.02%) |
| Runtime, fixed optimized Lib package | 16.889 ms | **16.007 ms** | 0.882 ms (5.22%) |

The second control includes dynamic-runtime experiments on top of #3234. Both runtimes replayed the same 158 kernel registrations, orchestration and input. The calendar experiment ran A/B/B/A on the same two devices. The analysis takes the first 20 timed rounds per phase after five warmups, giving 40 samples per runtime. The two paired reductions were 0.884 and 0.878 ms. All four phases returned the same 28 host-visible tensors bit for bit. This checks differential agreement on the frozen fixture. The measurements establish aggregate effects; individual operator contributions and the incremental Lib gain on a fixed calendar runtime still require separate comparisons.

## Source and replay provenance

The build used PyPTO `22371da7499c9318845e88f4d839a68df270cb9e`, PTOAS 0.78, PTO ISA `45435c0624c1d0d056bf38aa60d4f56dc0ef3b6b` and CANN 9.3. The source snapshot was rebuilt off-device and loaded through the production prebuilt loader. All 158 registrations matched in metadata, including IDs, signatures and progress declarations. Host orchestration matched, as did all 23 loadable sections of the device orchestration image.

Of the 158 registered device binaries, 153 matched the measured package byte for byte. The run used one frozen Indexer binary (SHA-256 `27d04a17589c55317e8502cf975687dcc6efa27da31def7bc9c8335a0b914355`) for five registrations: `_score_prepare`, `_score_aiv`, `_deepseek_select`, `_score_candidates__3_aiv` and `_deepseek_select_with_pool__2`. Repacking with those measured registrations and the measured device orchestration image reproduced the loader payload exactly. **16.007 ms is the measurement for that frozen package. A fresh source-only build of this branch has not been timed.**

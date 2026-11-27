# Tuning and validation notes

## Pipeline and sampling

For the 40-layer Flash-1017 variant, order visible GPUs as two NUMA-local groups of four. Apply the matching patch, then use `VLLM_PP_LAYER_PARTITION=20,20`, `--tensor-parallel-size 4`, `--pipeline-parallel-size 2`, and `--distributed-executor-backend mp`. Leave attention backend and KV dtype on auto unless there is a tested reason to change them. Keep backend topology checks in place: PCIe-only connectivity does not imply eligibility for NVLink-specific collectives.

The tested sampling profile used `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block"}`. For a fair TP8/PP2 comparison, keep front end, sampling, graph mode, token budget, prefix caching and prompt/output sizes constant. Use your own model path, bind address, and resource budgets. Start on loopback with a short context, confirm the expected backend and outputs, and raise context limits only after checking memory and evaluating long inputs.

## Optional: CPU KV lazy offload

This is an existing vLLM feature, not a change in this patch. It retains reusable prefix blocks near GPU eviction and reloads them on a hit. It does not keep an active decode sequence in CPU memory or directly increase the active GPU KV pool. For a local CPU-only tier, pass an explicit connector, for example:

```sh
--kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{"kv_offload_backend":"cpu","lazy_offload":true,"cpu_bytes_to_use":274877906944}}'
```

**256 GiB (274,877,906,944 bytes)** is the recommended total budget for the documented eight-GPU profile on a host with sufficient pinned-memory headroom; it is not a universal default or an allocation per GPU. Adjust it for your expected load and available RAM. Check host and per-NUMA headroom before using this budget; use a smaller budget if memory is tight. The two stages use the same slot count, sized for the larger block; the stage with smaller blocks may allocate less than its budget. Check the scheduler and all workers report a matching nonzero slot count and the CPU lazy backend.

The pinned vLLM revision rejects KV connectors when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` without CuMem. One supported option is to set `expandable_segments:False` **for the serving process only**, preserving any other allocator options and checking the effective `PYTORCH_ALLOC_CONF` alias. Do not bypass the validation. Recheck GPU KV capacity and graph capture after changing allocator settings. CPU cache startup was verified separately; the before/after throughput impact of this feature was not measured.

## Selected measurements

One paired comparison used eight A800 PCIe GPUs on two NUMA nodes, with no NVLink. The baseline used TP8 × PP1 with no explicit layer partition; the candidate used the patch and TP4 × PP2 with the `20,20` partition. Both sides used the same installed binaries, Rust streaming frontend, real DSpark K5 profile above, Engram CPU offload, prefix caching, and `FULL_AND_PIECEWISE` CUDA graphs. These are benchmark settings, not deployment defaults. For each workload there were eight serialized requests with short prompts and 512 generated tokens. The pooled decode rate excludes prefill and the first output block.

| Measurement | TP8 | TP4 × PP2 | Change |
| --- | ---: | ---: | ---: |
| Code decode (tok/s) | ~251 | ~320 | +28% |
| Prose decode (tok/s) | ~148 | ~207 | +40% |
| GPU KV capacity at startup (tokens) | ~21.3M | ~31.3M | +47% |

These are small-sample, single-stream results, not a claim about multi-user throughput, a one-million-token request, or output parity. The code rate is not the upstream PR's AgentX result. Benchmark synthetic acceptance is not a production alternative to real verification. For follow-up evaluations, measure cold prefill and TTFT separately from steady decode, then measure both again with mixed traffic. For CPU KV, track CPU reloads, locally computed prompt tokens, and TTFT before claiming a benefit.

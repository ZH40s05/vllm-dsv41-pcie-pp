# DeepSeek V4.1 Flash on PCIe Ampere: TP4 × PP2 patch kit

A Python-only vLLM adaptation for DeepSeek V4.1 Flash on an eight-GPU, dual-NUMA PCIe system. It reuses a compatible vLLM binary installation; the patch does not change CUDA kernels, the Rust frontend, weights, or the KV format. This is a standalone patch kit, not an official vLLM release.

## What the patch changes

- Allow DSpark auxiliary hidden states to cross a pipeline boundary, and retain the necessary local vocabulary embedding shard for the drafter.
- Capture only auxiliary taps owned by the local stage.
- Disable decoder-side SWA bounded replay when the final KV source is the stage's first layer and there is no local predecessor for replay. This does not turn off prefix caching.
- At the `20,20` split for the 40-layer Flash-1017 checkpoint, each stage owns its relevant KV source. Put each group of four tensor-parallel ranks on the same NUMA side; do not assume a different split has the same property.

## Apply against the tested base

Use a clean checkout of [vLLM at `b935849ce6450a9e660c04d6e41c83dc42a603c2`](https://github.com/vllm-project/vllm/commit/b935849ce6450a9e660c04d6e41c83dc42a603c2), a nightly after [#66497](https://github.com/vllm-project/vllm/pull/66497). Set `PATCH_KIT` to the absolute path of this repository. From the vLLM checkout:

```sh
git switch --detach b935849ce6450a9e660c04d6e41c83dc42a603c2
git apply --check "$PATCH_KIT/patches/vllm-b935849ce-dsv41-tp4pp2.patch"
git apply "$PATCH_KIT/patches/vllm-b935849ce-dsv41-tp4pp2.patch"
```

Run the modified Python package with a matching native build in an isolated environment; confirm both the launcher and worker imports see the patched modules. Do not overwrite an existing production installation. The included regression cases are in `tests/models/test_deepseek_v41_pipeline.py` in the patched checkout. Activate your matching vLLM test environment, then run from the patched checkout:

```sh
python -m pytest -q tests/models/test_deepseek_v41_pipeline.py
```

For this snapshot, apply and reverse-apply checks ran against files freshly exported from the pinned commit. The three patched Python files compiled, and the four partition-boundary cases passed in an isolated source-function check. The full CUDA pytest suite was not rerun for this packaging step.

See the [tuning and validation notes](docs/tuning.md) for topology, an example CPU KV configuration, and measurement limits.

## License and attribution

[Apache License 2.0](LICENSE). The patch modifies code from the [vLLM project](https://github.com/vllm-project/vllm) and preserves its existing attribution and SPDX headers. The changes are separate from the upstream PR and were prepared with AI assistance.

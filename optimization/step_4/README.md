step3 results showed that numerous runtime accessor resolutions in compute_snr_and_apply_filters were hitting performance.

(TODO: put actual figures from profiling)

Hypothesis: replacing the dual level loop that call operator() on tensors with a one level loop that iterates on n_y * n_x and uses raw pointers for accesses will remove these.

After a first try at applying this fix, we didn't see any effect on execution time. THe samples with runtime accessor solving disappeared but they were likely showing memory access times and nothing else.

## Reproducing

`perf-step_4-after` is a tag on top of `282e669` (the raw-pointer rewrite).

```bash
git worktree add --detach /tmp/step_4-after perf-step_4-after
nix develop --command meson setup /tmp/step_4-after/builddir /tmp/step_4-after
nix develop --command ninja -C /tmp/step_4-after/builddir bench
```

Hotspots profile (job 66349239, null result): `sbatch run_profile.sh 72`

Memory-access profile (job 66356748, NUMA findings): `sbatch run_bench_memaccess.sh`

## Before/after (bench's own timer)

Ran `bench <manifest> 72 5` directly (on an interactive compute-node allocation, no VTune
overhead) against the step 3 final state (`4a1a6c0`, "before") and this step's tag
(`perf-step_4-after` / `282e669`, "after"):

| | before (4a1a6c0) | after (282e669) |
|---|---:|---:|
| mean iteration (`run_batch_core`, 5 iters) | 12.463 s | 12.664 s (+1.6 %) |
| per-item cost, single-threaded (median) | 0.3963 s | 0.3630 s (-8.4 %) |

Confirms the null result from the hotspots profile: the raw-pointer rewrite measurably cuts
single-threaded per-item cost, but that doesn't translate into a faster multi-threaded
`run_batch_core`.
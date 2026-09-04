"after" profiling from step_2 showed a lot of time spent in xtensor accessor.
Asking Claude Code to identify where these accessors originated saved a bit of time in VTUne's interface that didn't show it properly (unwinding verbose C++ call stacks).
We were then able to identify that one line of xtensor calls wasn't unwinded and created lazy evaluations that were solved inside the loop, thus creating a large overhead: in step_2's profile (job 66346507, 72 threads, 5 iters), the `combine()` lambda's two `xt::noalias(...) = xt::sqrt(...)` lines were responsible for **613.9 s of CPU time (15.3 % of the top-100 hotspot total)** across `linear_assigner`, `xt::sqrt`, `detail::multiplies`/`detail::plus`, and the `xfunction` stepper machinery evaluating that expression tree. Overall, xtensor accessor/expression machinery across the whole binary was **60.7 %** of CPU time -- so this one call site was about a quarter of that total.

Hypothesis: manually unwinding these calls should reduce these overheads and lead to better performance.

## Result

Rewrote the two lines as an explicit `(i_y, i_x)` double loop over raw element access
(commit `ab698a7`, tagged `perf-step_3-after`). Two real bugs surfaced along the way that
the lazy-expression version was handling for free and had to be restated explicitly:

- **Shape checking.** `xt::sqrt(a + scale * group.selfnoise_offset + ...)` throws a catchable
  "incompatible dimension" error when `selfnoise_offset` doesn't match the plane's shape --
  `run_batch_core`'s per-item `try`/`catch` relies on exactly this to isolate one malformed item
  without crashing the batch (a test exercises this deliberately). The raw loop has no such
  check by construction, so it now does one `shape() != {n_y, n_x}` comparison per `combine()`
  call (not per pixel) and throws explicitly if it fails.
- **Implicit resize.** `xt::noalias(noisemap) = expr` auto-resizes `noisemap` to the RHS shape;
  the raw loop doesn't, so `noisemap`/`excess_noise_factor` needed an explicit `.resize({n_y, n_x})`
  before the loop -- missing it segfaulted on the first call (empty scratch buffer).

Re-profiled after the fix (job 66347490, same config as 66346507 -- 72 threads, 5 iters,
`lot_diff36_cpp`). Full report in `results/vtune-after-rewrite-job66347490.{summary.txt,hotspots.csv}`.

| | before (66346507) | after (66347490) |
|---|---:|---:|
| mean iteration (bench's own timer) | 13.238 s | 14.254 s |
| `combine()`-attributable CPU time (`linear_assigner`+`multiplies`+`plus`+`xfunction`+`sqrt`) | 613.9 s (15.3 %) | 443.7 s (10.9 %) |
| `linear_assigner` alone | 338.8 s (8.4 %) | 305.5 s (7.5 %) |
| `detail::multiplies` (elementwise scalar\*tensor) | 162.8 s | 0.0 s |
| all xtensor accessor/expression machinery (whole binary) | 60.7 % | 60.2 % |

**Hypothesis: partially confirmed, inconclusive at the wall-clock level.** The targeted category
did shrink -- the two lines' own overhead dropped by about a quarter (613.9 s → 443.7 s), and the
`multiplies` functor calls they drove disappeared entirely, exactly as expected from removing the
lazy-expression evaluation. But this was only ~4 points off a 60 %-plus total dominated by many
*other* xtensor call sites (`xt::median`/`nth_element` sorting alone is still ~23 %, plus several
other `xt::view`/`xt::any`/`xt::sum` call sites elsewhere in `process_band` untouched by this
change), so it didn't move the needle on overall CPU-time share. The mean iteration time actually
came out slightly slower (14.254 s vs 13.238 s) -- most likely noise given the run-to-run variance
already documented on this cluster (step_1's README: three runs of nominally the same code gave
14.8/25.0/19.8 s/iter), but this is a single sample each side and shouldn't be read as a real
regression without repeat runs (`iterations=3-5` per side) to separate signal from node-to-node
noise.

**Implication for the broader "convert every accessor to raw pointers" idea**: this result argues
against it. The overhead is spread thin across many call sites, not concentrated in one; each
individual rewrite carries real correctness risk (as this one did, twice) for a few points of one
metric. A worthwhile next step would be to profile which of the *other* xtensor call sites in
`process_band` (the `xt::view`/`xt::any`/`xt::sum` usages, or the `xt::abs(plane)` branch) carry
the largest remaining share, and decide case by case -- rather than a blanket rewrite.

## Reproducing

`perf-step_3-after` is a real tag on top of `a303c96` (the build-config fix, see step 1's
README), so it configures without needing `build-fix.patch`:

```bash
git worktree add --detach /tmp/step_3-after perf-step_3-after
nix develop --command meson setup /tmp/step_3-after/builddir /tmp/step_3-after
nix develop --command ninja -C /tmp/step_3-after/builddir bench
sbatch ../step_1/vtune_job.sh /tmp/step_3-after/builddir/bench step_3-after 72 5
```

Issue: Claude Code made a mistake and attributed all the accessor overhead to ONE CALL, which turned out to be wrong and it then moved to pointing out call to xt::median.
Next step is to identify all the situations where basic strided accessors are used to get rid of them when possible.
- xt::median also uses lazy xtensor calls and switching to std::nth_elements on raw pointer should solve the issue.

## Result 2: `xt::median` bypass

First attempt: materialize `xt::abs(plane)` into a concrete tensor before calling `xt::median`,
so it would see a contiguous, statically-laid-out container instead of a lazy expression
(job 66349086). This backfired as a diagnostic more than a fix: it made `xt::median` itself
show up as the single largest hotspot in the whole binary (616.8 s, 14.9 % of CPU time) instead
of reducing anything -- `xt::median` carries real overhead of its own, independent of whether
its input is lazy or already concrete. Previously that same cost was just smeared thin across
many differently-instantiated symbols (`nth_element<xiterator<...>>`, `xtensor_container`
construction, `xfunction_stepper`), each individually too small to show up.

Actual fix: replaced `xt::median` outright, at the three call sites that operate on plain
contiguous tensors (`plane`, `windows_sums`, `noisemap`), with a hand-rolled
`std::nth_element`-based median over a reused scratch buffer
(`detail::median_of`/`median_of_abs`, `ProcessBandScratch::median_scratch`). Re-profiled
(job 66349126, same config -- 72 threads, 5 iters, `lot_diff36_cpp`):

| | before (66347490) | after (66349126) |
|---|---:|---:|
| mean iteration (bench's own timer) | 14.254 s | **11.626 s (-18.4 %)** |
| total CPU time | 4059.6 s | 3798.6 s (-6.4 %) |
| `xt::median` as a top-5 hotspot | not present (smeared into "Others") | gone entirely |

Good performance improvement overall.
See `results/vtune-nth-element-median-job66349126.{out,provenance.txt,source.patch}`.

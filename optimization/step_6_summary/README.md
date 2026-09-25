
before	after	change
mean iteration (run_batch_core)	24.916 s	11.268 s	-54.8% (2.21×)
per-iteration range	24.255~25.649 s	11.035~11.454 s	
per-item cost, single-threaded (mean)	0.6118 s	0.3721 s	-39.2%

## Thread-count scaling

| threads | before (s) | before speedup | before eff. | after (s) | after speedup | after eff. | after vs before |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 664.797 | 1.00x | 100% | 400.871 | 1.00x | 100% | 1.66x |
| 2 | 335.911 | 1.98x | 99.0% | 202.106 | 1.98x | 99.2% | 1.66x |
| 4 | 169.236 | 3.93x | 98.2% | 103.134 | 3.89x | 97.2% | 1.64x |
| 9 | 76.612 | 8.68x | 96.4% | 50.315 | 7.97x | 88.5% | 1.52x |
| 18 | 41.146 | 16.16x | 89.8% | 23.785 | 16.86x | 93.6% | 1.73x |
| 24 | 32.857 | 20.23x | 84.3% | 19.143 | 20.94x | 87.3% | 1.72x |
| 36 | 27.214 | 24.43x | 67.9% | 14.943 | 26.83x | 74.5% | 1.82x |
| 48 | 26.624 | 24.97x | 52.0% | 12.092 | 33.16x | 69.1% | 2.20x |
| 64 | 25.207 | 26.37x | 41.2% | 11.244 | 35.66x | 55.7% | 2.24x |
| 72 | 24.594 | 27.03x | 37.5% | 11.016 | 36.40x | 50.6% | 2.23x |

64/72 "after" are 8-iteration means (job 66555241) -- the original single-iteration values
(11.374s / 12.047s) made 72 look slower than 64, which didn't hold up under repeat measurement.

Both versions scale near-linearly to ~9-18 threads, then flatten as the workload saturates DRAM
bandwidth (consistent with step 5's memory-bound finding). The optimized version holds
meaningfully better parallel efficiency at every thread count >=18, and its speedup over the
unoptimized version *grows* with thread count (1.5-1.7x at low counts, up to 2.2x at 48-72
threads) -- the raw-pointer rewrite scales better, not just a constant per-item overhead cut.

## Memory-access profiling across thread counts

| threads | Memory Bound (% pipeline slots) | DRAM avg GB/s (aggregate) | DRAM obs. max GB/s (aggregate) | DRAM % time in high-BW band | UPI Util. Bound (% elapsed) |
|---:|---:|---:|---:|---:|---:|
| 9 | 28.7% | 32.3 | 77.4 | 0.0% | 0.0% |
| 18 | 86.7% | 24.7 | 99.9 | 0.0% | 0.0% |
| 36 | 63.8% | 38.3 | 222.9 | 2.1% | 20.1% |
| 48 | 80.6% | 41.1 | 257.4 | 21.7% | 15.8% |
| 64 | 79.4% | 46.4 | 315.5 | 20.8% | 19.0% |
| 72 | 89.8% | 45.8 | 311.2 | 19.6% | 19.0% |

Platform DRAM ceiling: 178 GB/s aggregate. High-BW-band time jumps from ~0-2% to ~20% right
where wall-clock scaling flattens (36->48 threads), but stays at ~20%. UPI Bound is comparably
large. The scaling plateau is likely due to a mix of DRAM bandwidth pressure and cross-socket memory traffic.

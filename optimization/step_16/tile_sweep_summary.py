#!/usr/bin/env python3
"""Summarize a run_tile_sweep.sh log: per-item single-thread mean and T=72 mean (iteration 0 dropped) per tile size."""
import re
import statistics as s
import sys

item, iters = {}, {}
for line in open(sys.argv[1]):
    m = re.match(r'round\d (\S+) per-item cost.*mean=([\d.]+)s', line)
    if m:
        item.setdefault(m[1], []).append(float(m[2]))
    m = re.match(r'round\d (\S+)\s+iteration (\d+): .*= ([\d.]+)s', line)
    if m and m[2] != '0':
        iters.setdefault(m[1], []).append(float(m[3]))

flat_item, flat_iter = s.mean(item['flat']), s.mean(iters['flat'])
print('| tile | one thread, per item | vs flat | T=72 mean | vs flat | n |')
print('|---|---:|---:|---:|---:|---:|')
for label in sorted(iters, key=lambda k: (k != 'flat', int(k.split('=')[1]) if '=' in k else 0)):
    i, t = s.mean(item[label]), iters[label]
    print(f"| {label} | {i:.4f} s | {100 * (i / flat_item - 1):+.1f}% | {s.mean(t):.2f} +- {s.stdev(t):.2f} s | "
          f"{100 * (s.mean(t) / flat_iter - 1):+.1f}% | {len(t)} |")

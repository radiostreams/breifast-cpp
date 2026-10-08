#!/usr/bin/env python3
"""Summarize a run_ab.sh log: single-thread per-item mean, then T18-bound and T72-single means (iteration 0 dropped)."""
import re
import statistics as s
import sys

times, detections, per_item = {}, {}, {}
for line in open(sys.argv[1]):
    m = re.match(r'(\w+) (\S+)\s+iteration (\d+): \d+ items, (\d+) detections.*= ([\d.]+)s', line)
    if m:
        detections.setdefault((m[2], m[1]), set()).add(m[4])
        if m[3] != '0':
            times.setdefault((m[2], m[1]), []).append(float(m[5]))
    m = re.match(r'(\w+) \S+ per-item cost.*mean=([\d.]+)s', line)
    if m:
        per_item.setdefault(m[1], []).append(float(m[2]))

if per_item:
    b, a = s.mean(per_item['before']), s.mean(per_item['after'])
    print(f"| one thread, per item | {b:.4f} s | {a:.4f} s | {100 * (a / b - 1):+.1f}% |")
for config in ['T18-bound', 'T72-single']:
    b, a = times[(config, 'before')], times[(config, 'after')]
    print(f"| {config} | {s.mean(b):.3f} +- {s.stdev(b):.3f} s | {s.mean(a):.3f} +- {s.stdev(a):.3f} s | "
          f"{100 * (s.mean(a) / s.mean(b) - 1):+.1f}% | detections {sorted(detections[(config, 'before')])} "
          f"{sorted(detections[(config, 'after')])}")

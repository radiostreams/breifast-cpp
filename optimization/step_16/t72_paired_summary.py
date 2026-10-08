#!/usr/bin/env python3
"""Summarize run_t72_paired.sh: per-invocation means (iteration 0 dropped), pooled stats and the paired difference per round."""
import math
import re
import statistics as s
import sys

times = {}
for line in open(sys.argv[1]):
    m = re.match(r'round(\d+) (\w+)\s+iteration (\d+): .*= ([\d.]+)s', line)
    if m and m[3] != '0':
        times.setdefault((int(m[1]), m[2]), []).append(float(m[4]))

rounds = sorted({r for r, _ in times})
for label in ('flat', 'tile'):
    pooled = [t for r in rounds for t in times.get((r, label), [])]
    means = [s.mean(times[(r, label)]) for r in rounds if (r, label) in times]
    print(f"{label}: {len(pooled)} iterations, mean {s.mean(pooled):.3f} s, sd {s.stdev(pooled):.3f} s; "
          f"invocation means {[round(m, 2) for m in means]}")

diffs = [s.mean(times[(r, 'tile')]) - s.mean(times[(r, 'flat')]) for r in rounds if (r, 'tile') in times and (r, 'flat') in times]
flat_mean = s.mean([s.mean(times[(r, 'flat')]) for r in rounds if (r, 'flat') in times])
md, sd = s.mean(diffs), s.stdev(diffs)
se = sd / math.sqrt(len(diffs))
print(f"paired tile - flat per round: {[round(d, 2) for d in diffs]}")
print(f"mean {md:+.3f} s ({100 * md / flat_mean:+.1f}% of flat), sd {sd:.3f} s, se {se:.3f} s, t = {md / se:.2f} (n = {len(diffs)} rounds)")

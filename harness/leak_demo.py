"""The normalise-before-mask leak, on synthetic data only (no dataset, no model, a few seconds on a laptop).
If a window is standardised over ALL its points and one point is then hidden, the visible scaled values sum to minus the hidden one,
so the 'above or below the left neighbour' answer can be read off the visible values exactly. Standardising over the visible points
only removes it. For a random walk the best honest answer is the neighbour rule, right with probability 1 - arccos(sqrt(k/(k+m)))/pi.
usage: PYTHONPATH=. python leak_demo.py"""
import numpy as np
from lateral_task import L, QPOS, STAGES, gen_dose
rng = np.random.default_rng(0); walks = [gen_dose(rng, 1.0) for _ in range(20000)]
print(f"{'scale':>6} {'hidden':>6} {'k,m':>5} {'closed form':>11} {'neighbour rule':>14} {'leak reader, all-point scaling':>30} {'same reader, visible-only scaling':>33}")
for st in range(3):
    i = QPOS[st]; shown = sorted(set(STAGES[st]) - {i}); left = max(j for j in shown if j < i); right = min(j for j in shown if j > i); k, m = i - left, right - i
    y = np.array([x[i] > x[left] for x in walks]); rule = np.array([x[right] > x[left] for x in walks])
    def leak(mode):
        out = []
        for x in walks:
            ref = x if mode == "all" else x[shown]; z = (x - ref.mean()) / (ref.std() + 1e-9); hidden_mean = -z[shown].sum() / (L - len(shown)); out.append(hidden_mean > z[left])
        return np.array(out)
    print(f"{st+1:>6} {L-len(shown):>6} {k},{m:>3} {1 - np.arccos(np.sqrt(k/(k+m)))/np.pi:>11.3f} {(rule == y).mean():>14.3f} {(leak('all') == y).mean():>30.3f} {(leak('visible') == y).mean():>33.3f}")

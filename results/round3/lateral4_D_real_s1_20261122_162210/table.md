# curriculum arm D / real / train seed 1: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [[(2, 1.0)], [(1, 1.0)], [(0, 1.0)]]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.490 | 0.490 | 0.494 |
| after_phase1 | 0.604 | 0.620 | 0.697 |
| after_phase2 | 0.543 | 0.759 | 0.621 |
| after_phase3 | 0.714 | 0.737 | 0.626 |
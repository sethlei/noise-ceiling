# curriculum arm C / real / train seed 1: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [None, None, None]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.490 | 0.490 | 0.494 |
| after_phase1 | 0.666 | 0.695 | 0.602 |
| after_phase2 | 0.705 | 0.758 | 0.640 |
| after_phase3 | 0.727 | 0.742 | 0.636 |
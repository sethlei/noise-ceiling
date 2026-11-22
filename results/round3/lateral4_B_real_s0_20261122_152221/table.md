# curriculum arm B / real / train seed 0: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [[(0, 1.0)], [(1, 0.6666666666666666), (0, 0.3333333333333333)], [(2, 0.5), (1, 0.25), (0, 0.25)]]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.490 | 0.490 | 0.494 |
| after_phase1 | 0.711 | 0.618 | 0.544 |
| after_phase2 | 0.737 | 0.778 | 0.557 |
| after_phase3 | 0.722 | 0.758 | 0.647 |
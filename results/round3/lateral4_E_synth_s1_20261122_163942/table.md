# curriculum arm E / synth / train seed 1: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [[(2, 1.0)], [(2, 1.0)], [(2, 1.0)]]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.507 | 0.503 | 0.483 |
| after_phase1 | 0.607 | 0.643 | 0.867 |
| after_phase2 | 0.600 | 0.647 | 0.900 |
| after_phase3 | 0.703 | 0.653 | 0.913 |
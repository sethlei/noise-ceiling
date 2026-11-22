# curriculum arm C / synth / train seed 1: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [None, None, None]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.507 | 0.503 | 0.483 |
| after_phase1 | 0.790 | 0.870 | 0.797 |
| after_phase2 | 0.813 | 0.880 | 0.853 |
| after_phase3 | 0.823 | 0.890 | 0.863 |
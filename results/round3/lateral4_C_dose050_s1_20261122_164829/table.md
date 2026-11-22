# curriculum arm C / dose / train seed 1: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = [None, None, None]
| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |
|---|---|---|---|
| nothing | 0.497 | 0.473 | 0.513 |
| after_phase1 | 0.715 | 0.842 | 0.817 |
| after_phase2 | 0.740 | 0.850 | 0.803 |
| after_phase3 | 0.752 | 0.860 | 0.818 |
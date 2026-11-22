# Round 3 results against the stakes (2026-11-22, Daniel; spec and amendment `ROUND3_SPEC_2026-11-22.md`)

24 H100 runs, none failed (batch 15:13–17:18 UTC, ≈ $9.50). Receipts and per-item predictions: `results/round3/`. Ceilings and
scoring: `harness/ceiling4.py` → `results/round3/ceiling4_scored.json` / `.txt`. All lateral cells use the leak-free scaling; the
real cells hold out 16 series and drop their 8 twins from training (623 eval items: 600 held-out windows + 23 WTI windows).

## The task and its ceiling
A 36-point window, some points hidden; is the hidden point above or below the nearest visible point to its left? Three scales:
S1 (left neighbour 3 steps away, right 3), S2 (3 and 1), S3 (1 and 1). For a random walk the best possible reader is the
neighbour rule (above iff the right neighbour is above the left), right with probability 1 − arccos(√(k/(k+m)))/π.

| | S1 | S2 | S3 |
|---|---|---|---|
| random walk, closed form | 0.750 | 0.833 | 0.750 |
| neighbour rule, all 6,722 real windows (109 series) | 0.742 | 0.820 | 0.741 |
| neighbour rule, the 623 held-out eval items | 0.740 | 0.815 | 0.727 |
| logistic / boosted trees trained on 1,500 windows, same items | 0.717 / 0.724 | 0.791 / 0.809 | 0.705 / 0.714 |

On held-out real series nothing trained beats the rule, and the rule sits on the walk's closed form. For this question the real
monthly series are a random walk, and the ceiling is known in advance.

## The LM on real windows (top-1, mean of two seeds; seeds agree within 0.03 in every cell)

| arm | S1 | S2 | S3 | S3, items where S2's label disagrees (n = 322; rule 0.739) |
|---|---|---|---|---|
| A staged coarse→fine | 0.615 | 0.620 | **0.708** | 0.705 |
| B staged + replay | 0.713 | **0.744** | 0.647 | 0.550 |
| C mixed | **0.722** | 0.737 | 0.649 | 0.576 |
| D staged fine→coarse | 0.710 | 0.740 | 0.610 | 0.491 |
| E fine only | 0.563 | 0.583 | 0.670 | 0.675 |
| ceiling (rule) | 0.740 | 0.815 | 0.727 | 0.739 |

## The noise dose (arm C; d = the random walk's share of the one-step variance; ceiling = boosted trees, 50,000 train / 20,000 eval)

| d | S1 LM / ceiling | S2 LM / ceiling | S3 LM / ceiling |
|---|---|---|---|
| 0 (clean sine) | 0.832 / 0.914 | 0.985 / 0.998 | 0.963 / 0.997 |
| 0.10 | 0.807 / 0.838 | 0.968 / 0.975 | 0.950 / 0.953 |
| 0.25 | 0.782 / 0.809 | 0.918 / 0.957 | 0.898 / 0.912 |
| 0.50 (two seeds) | 0.742 / 0.780 | 0.861 / 0.923 | 0.824 / 0.850 |
| 0.75 | 0.720 / 0.759 | 0.845 / 0.875 | 0.755 / 0.791 |
| 1 (pure walk, two seeds) | 0.736 / 0.758 | 0.774 / 0.830 | 0.694 / 0.747 |

The LM follows the ceiling down the whole curve, 0.003–0.083 under it (median shortfall 0.033 over the 18 cells). As the walk's
share rises the ceiling falls from ≈ 1 to the closed form and the LM falls with it; no training choice here moves the ceiling.

## Stakes
- **K1 (the LM never used the leak; arm E real S3 within 0.05 of the rule on the same items): FAIL by 0.007.** E = 0.670 against the
  rule's 0.727, 0.057 under; not the 0.08 that would say the old number leaned on the leak. The leaking instrument's 0.735 was at
  most 0.03–0.06 generous (it also had the eval series in training). Arm A, which ends on the fine scale, reaches 0.708.
- **K2 (the S2 gap is real; best arm mean ≥ 0.04 under the rule): PASS.** B = 0.744 against 0.815, 0.071 under. The same gap shows
  on the pure synthetic walk (C: 0.774 against 0.830), so it is not a property of the real data. Round 3b asks whether it is a
  shortage of training on that scale.
- **K3 (disagreement-set ordering replicates; E beats B and C by ≥ 0.05 on the real S3 disagree set, both seeds): PASS.** E 0.675
  (0.674, 0.677); B 0.550 (0.550, 0.550); C 0.576 (0.568, 0.584). The multi-scale readers answer the coarser question on the
  items where the two scales differ: on those items B and C are barely above a coin while the rule is at 0.739. The 11-20
  finding survives the clean instrument, larger than before.
- **K4 (the LM tracks the ceiling along the dose, within 0.06 except S2 at d ≥ 0.5): 16 of 18 cells.** Misses: d = 0 at S1
  (0.082 under: a clean sine at the coarse scale, where the rule is useless and the reader must fit the period) and d = 0.5 at S2
  (0.062, inside the 0.04–0.10 I expected there). At d = 0.75 the S2 gap is 0.030, smaller than I staked.
- **K5 (real series sit at the walk end; best arm within 0.05 of the d = 1 row): PASS.** S1 0.722 against 0.736; S3 0.708 against 0.694.
- **K6 (key-free forecast, restated: T1 against Gaussian climatology ≤ +0.07 and output variation > 0.01 at the selected epoch):
  FAIL on both.** +0.237 [+0.141, +0.329]; output variation 0.0007. What it does show: with months, prices and window statistics
  out of the prompt, the WTI training loss never leaves ln 7 (1.946–1.976 over twelve epochs; the keyed run fell to 1.58 while
  validation rose to 2.29). The keyed run's learning was the keys. Without them this recipe fits nothing on the forecast task; the
  score is a constant answer sharpened by the temperature fit (T = 0.595) onto bins the test years did not favour.

## Reading
1. **The stochasticity sets a ceiling that can be written down.** For the hidden-point question on monthly economic series it is the
   random walk's 0.75 / 0.83 / 0.75. A simple rule reaches it; logistic regression and boosted trees do not pass it; the LM's best
   arms are 0.02 under it at two scales and 0.07 under at the third.
2. **Three instrument faults produced numbers above that ceiling, and each looked like skill.** The scaling leak (0.94 available
   from the sum of the visible points); eval series present in training, with twins (WTI under another name); keys in the prompt
   (training loss falls, validation rises). A score above the walk's ceiling on this kind of data is a reason to look for a leak.
3. **Curriculum structure changes which items are right, not how many.** Every arm that trains all scales lands near the same
   headline and gets there by answering the coarse question; on the items where scales disagree it is near chance. Fine-only and
   coarse-to-fine keep the fine reading and lose the coarse scales.
4. **Reading is not the forecast.** The same model reads the sign of a shown 12-month move at 0.98 (11-19) and fits nothing on the
   forecast task once the keys are gone. The forecast target on these series has no ceiling above climatology that this recipe
   finds; the linear momentum column's −0.037 remains the only positive result.

## Round 3b: the middle-scale gap (arm F, S2 only; one seed; four runs, ≈ $3; receipts `results/round3/lateral4_F_*`)

| source | training sequences | S2 after 3 / 6 / 9 epochs | ceiling |
|---|---|---|---|
| pure walk (d = 1) | 1,500 | 0.813 / 0.820 / 0.830 | 0.830 |
| pure walk (d = 1) | 6,000 | 0.827 / 0.832 / 0.828 | 0.830 |
| real, held-out series | 1,500 | 0.801 / 0.791 / 0.811 | 0.815 |
| real, held-out series | 4,500 | 0.809 / 0.791 / 0.788 | 0.815 |

- **K7 (S2-only training at the larger size reaches within 0.03 of the rule on both): PASS** (0.828 against 0.830; 0.788 against 0.815,
  0.027 under), and the premise of the stake was off: the reader is already at the ceiling at 1,500 sequences (0.830, 0.811). More
  data did nothing; on real it ended lower (0.788), within one seed's noise (SE ≈ 0.016).
- So the 0.07 gap in round 3 was not the data and not a limit of the reader. It is the cost of training several scales in one
  adapter: mixed training gives up 0.08 at S2 and S3 against the rule (0.737, 0.649) and single-scale training gives up 0.004 and
  0.057 (F at S2; E at S3). At S1 mixed is as good as single-scale (0.722 against A's 0.709 after its S1 phase).
- One seed per cell here; the round-3 seeds agreed within 0.03 everywhere, which is the scale of what one seed can hide.

The H100 box was stopped at 18:00 UTC (verified STOPPED). Day's spend ≈ 2.9 h ≈ $13.

*Stakes: four passed (K2, K3, K5, K7), one mostly (K4, 16 of 18), two failed (K1 by 0.007; K6).*

# recall2 up/down over the last 12 months, generic windows, no keys in text
| eval set | n | top-1 | majority | log score | uniform | output variation | argmax counts |
|---|---|---|---|---|---|---|---|
| eval_other_series | 400 | 0.910 | 0.640 | 0.483 | 0.693 | 0.4266 | [274, 126] |
| eval_wti_2019_2026 | 94 | 0.979 | 0.553 | 0.058 | 0.693 | 0.4875 | [54, 40] |
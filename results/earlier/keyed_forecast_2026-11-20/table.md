| family | h | n | model RMSE | model MAE | reference RMSE | reference MAE | model Brier | uniform Brier | Gaussian clim. Brier | model log score | uniform log score | Gaussian clim. log score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1 | 48 | 6.24 | 5.20 | 5.71 | 4.66 | 0.901 | 0.857 | 0.840 | 2.100 | 1.946 | 1.885 |
| T1 | 3 | 48 | 11.50 | 9.25 | 10.69 | 7.92 | 0.896 | 0.857 | 0.862 | 2.079 | 1.946 | 1.941 |
| T1 | 6 | 48 | 16.81 | 14.00 | 15.03 | 10.92 | 0.896 | 0.857 | 0.851 | 2.109 | 1.946 | 1.910 |
| T1 | 12 | 46 | 22.10 | 18.89 | 18.93 | 14.92 | 0.886 | 0.857 | 0.793 | 2.072 | 1.946 | 1.763 |
| T2 | 1 | 57 | 4.42 | 3.30 | 4.31 | 3.17 | 0.861 | 0.857 | 0.874 | 1.944 | 1.946 | 2.004 |
| T2 | 3 | 55 | 12.79 | 9.82 | 12.72 | 9.41 | 0.867 | 0.857 | 0.840 | 1.989 | 1.946 | 1.879 |
| T2 | 6 | 52 | 16.82 | 13.44 | 16.52 | 12.22 | 0.867 | 0.857 | 0.863 | 2.004 | 1.946 | 1.936 |
| T2 | 12 | 46 | 20.18 | 17.43 | 19.84 | 14.88 | 0.866 | 0.857 | 0.778 | 2.010 | 1.946 | 1.719 |

Information set: per-signal; reference column = random walk (T1) / STEO (T2); uniform = 1/7 over the equal-mass training bins; Gaussian clim. = N(train residual mean, sd) through the bin edges, the ordinal null (the reference's own scores as a point mass are in table.json). USD per barrel on the WTI monthly mean; test split = target months 2022-01 to 2026-10; compare with `baselines_2026-11-11.md`.
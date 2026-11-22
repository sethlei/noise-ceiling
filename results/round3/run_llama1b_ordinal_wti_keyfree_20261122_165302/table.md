| family | h | n | model RMSE | model MAE | reference RMSE | reference MAE | model Brier | uniform Brier | Gaussian clim. Brier | model log score | uniform log score | Gaussian clim. log score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1 | 48 | 6.17 | 5.01 | 5.71 | 4.66 | 0.916 | 0.857 | 0.840 | 2.115 | 1.946 | 1.885 |
| T1 | 3 | 48 | 11.94 | 9.43 | 10.69 | 7.92 | 0.909 | 0.857 | 0.862 | 2.081 | 1.946 | 1.941 |
| T1 | 6 | 48 | 17.50 | 14.14 | 15.03 | 10.92 | 0.918 | 0.857 | 0.851 | 2.121 | 1.946 | 1.910 |
| T1 | 12 | 46 | 23.84 | 19.99 | 18.93 | 14.92 | 0.920 | 0.857 | 0.793 | 2.134 | 1.946 | 1.763 |
| T2 | 1 | 57 | 4.62 | 3.24 | 4.31 | 3.17 | 0.895 | 0.857 | 0.874 | 2.045 | 1.946 | 2.004 |
| T2 | 3 | 55 | 13.19 | 10.30 | 12.72 | 9.41 | 0.888 | 0.857 | 0.840 | 2.041 | 1.946 | 1.879 |
| T2 | 6 | 52 | 17.49 | 13.95 | 16.52 | 12.22 | 0.900 | 0.857 | 0.863 | 2.075 | 1.946 | 1.936 |
| T2 | 12 | 46 | 21.24 | 18.04 | 19.84 | 14.88 | 0.890 | 0.857 | 0.778 | 2.051 | 1.946 | 1.719 |

Information set: per-signal; reference column = random walk (T1) / STEO (T2); uniform = 1/7 over the equal-mass training bins; Gaussian clim. = N(train residual mean, sd) through the bin edges, the ordinal null (the reference's own scores as a point mass are in table.json). USD per barrel on the WTI monthly mean; test split = target months 2022-01 to 2026-10; compare with `baselines_2026-11-11.md`.
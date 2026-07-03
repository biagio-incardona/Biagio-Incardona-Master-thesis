# National ILI Benchmark Run Report

- **Period Covered:** 2003-10-19 to 2015-07-26
- **Number of Origins:** 113
- **Horizons:** [1, 2, 4, 8]
- **Completed Models:** TimeGPT
- **Failed Models:** 0

## Summary Metrics (MAE/WIS)

| model              |     MAE |      WIS |
|:-------------------|--------:|---------:|
| Chronos-V2         | 1.05752 | 0.705916 |
| TiRex              | 1.06657 | 0.737739 |
| TimesFM            | 1.10794 | 0.743654 |
| Chronos-Small      | 1.10994 | 0.816625 |
| Chronos-Bolt-small | 1.25168 | 0.889349 |
| CatBoost           | 1.33181 | 1.01614  |
| ARIMA              | 1.34044 | 0.914797 |
| SARIMA             | 1.36429 | 0.939731 |
| Ridge              | 1.37519 | 1.00624  |
| XGBoost            | 1.62948 | 1.2295   |
| TimeGPT            | 1.76931 | 1.2688   |
| LightGBM           | 1.78346 | 1.3257   |
| Naive              | 2.10681 | 1.48866  |
| Drift              | 2.12465 | 1.50148  |
| ETS                | 2.12603 | 1.35503  |
| MovingAverage      | 2.89174 | 1.9255   |
| Prophet            | 3.10902 | 2.04681  |
| SeasonalNaive      | 3.31583 | 2.30175  |
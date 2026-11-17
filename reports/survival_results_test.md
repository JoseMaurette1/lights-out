| model                          |   C-index |   Integrated Brier |   median abs error (h) |   within 2x of actual |
|:-------------------------------|----------:|-------------------:|-----------------------:|----------------------:|
| Random (no skill)              |     0.5   |            nan     |                nan     |               nan     |
| Cox PH                         |     0.688 |              0.16  |                nan     |               nan     |
| Random survival forest         |     0.672 |              0.157 |                nan     |               nan     |
| LightGBM (log hours)           |     0.711 |              0.151 |                  3.121 |                 0.575 |
| CatBoost (log hours)           |     0.711 |              0.151 |                  3.121 |                 0.575 |
| Ensemble (LightGBM + CatBoost) |     0.712 |              0.151 |                  3.115 |                 0.576 |
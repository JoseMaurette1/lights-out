| model                  |   C-index |   Integrated Brier |   median abs error (h) |   within 2x of actual |
|:-----------------------|----------:|-------------------:|-----------------------:|----------------------:|
| Random (no skill)      |     0.5   |            nan     |                nan     |               nan     |
| Cox PH                 |     0.678 |              0.163 |                nan     |               nan     |
| Random survival forest |     0.665 |              0.158 |                nan     |               nan     |
| LightGBM (log hours)   |     0.7   |              0.155 |                  3.277 |                 0.563 |
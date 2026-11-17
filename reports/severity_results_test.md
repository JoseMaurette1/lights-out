| model                                              |   RMSE (log) |   MAE (log) |   MAE (pct pts) |   macro F1 |   accuracy |   recall low |   recall medium |   recall high |
|:---------------------------------------------------|-------------:|------------:|----------------:|-----------:|-----------:|-------------:|----------------:|--------------:|
| Baseline (storm type)                              |        0.941 |       0.667 |           2.948 |      0.296 |      0.719 |        0.997 |           0     |         0.028 |
| Linear / logistic                                  |        0.848 |       0.587 |           2.784 |      0.519 |      0.643 |        0.696 |           0.481 |         0.58  |
| Random forest                                      |        0.829 |       0.551 |           2.714 |      0.546 |      0.736 |        0.864 |           0.458 |         0.271 |
| LightGBM                                           |        0.812 |       0.526 |           2.645 |      0.565 |      0.744 |        0.872 |           0.449 |         0.332 |
| CatBoost                                           |        0.814 |       0.531 |           2.648 |      0.568 |      0.743 |        0.869 |           0.45  |         0.35  |
| Ensemble (LightGBM + CatBoost)                     |        0.811 |       0.528 |           2.644 |      0.568 |      0.746 |        0.873 |           0.448 |         0.34  |
| Ensemble (LightGBM + CatBoost, F1-tuned cutoffs)   |      nan     |     nan     |         nan     |      0.565 |      0.748 |        0.887 |           0.405 |         0.36  |
| Ensemble (LightGBM + CatBoost, high-recall cutoff) |      nan     |     nan     |         nan     |      0.571 |      0.739 |        0.865 |           0.409 |         0.443 |
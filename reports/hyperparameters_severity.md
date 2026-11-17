| model                    | hyperparameter    | value                                                                    |
|:-------------------------|:------------------|:-------------------------------------------------------------------------|
| Baseline (storm type)    | rule              | mean (regression) / most common class per storm type                     |
| Ridge                    | alpha             | 1000.0                                                                   |
| Logistic                 | C                 | 10.0                                                                     |
| Logistic                 | class_weight      | balanced                                                                 |
| Logistic                 | max_iter          | 3000                                                                     |
| Random forest (both)     | n_estimators      | 300                                                                      |
| Random forest regressor  | max_features      | 0.5                                                                      |
| Random forest regressor  | min_samples_leaf  | 6                                                                        |
| Random forest regressor  | max_depth         | 20                                                                       |
| Random forest regressor  | max_samples       | 0.35                                                                     |
| Random forest classifier | max_features      | 0.2                                                                      |
| Random forest classifier | min_samples_leaf  | 7                                                                        |
| Random forest classifier | max_depth         |                                                                          |
| Random forest classifier | max_samples       | 0.78                                                                     |
| Random forest classifier | class_weight      | balanced_subsample                                                       |
| LightGBM regressor       | learning_rate     | 0.0111                                                                   |
| LightGBM regressor       | num_leaves        | 84                                                                       |
| LightGBM regressor       | min_child_samples | 115                                                                      |
| LightGBM regressor       | subsample         | 0.793                                                                    |
| LightGBM regressor       | colsample_bytree  | 0.519                                                                    |
| LightGBM regressor       | reg_lambda        | 0.0016                                                                   |
| LightGBM regressor       | min_split_gain    | 0.162                                                                    |
| LightGBM regressor       | cat_smooth        | 1.55                                                                     |
| LightGBM classifier      | learning_rate     | 0.0217                                                                   |
| LightGBM classifier      | num_leaves        | 15                                                                       |
| LightGBM classifier      | min_child_samples | 244                                                                      |
| LightGBM classifier      | subsample         | 0.541                                                                    |
| LightGBM classifier      | colsample_bytree  | 0.439                                                                    |
| LightGBM classifier      | reg_lambda        | 10.07                                                                    |
| LightGBM classifier      | min_split_gain    | 0.258                                                                    |
| LightGBM classifier      | cat_smooth        | 7.17                                                                     |
| CatBoost (both)          | learning_rate     | 0.06                                                                     |
| CatBoost (both)          | depth             | 7                                                                        |
| CatBoost (both)          | l2_leaf_reg       | 5                                                                        |
| CatBoost (both)          | random_seed       | 42                                                                       |
| CatBoost (both)          | iterations        | early stopping on 2020 (patience 100), scaled for the refit              |
| CatBoost classifier      | class_weights     | sqrt_balanced                                                            |
| Ensemble                 | rule              | average of LightGBM and CatBoost predictions (probabilities for classes) |
| LightGBM classifier      | class_weight      | sqrt_balanced                                                            |
| LightGBM (both)          | n_estimators      | early stopping on 2020 (patience 100), x1.21 for the 2014-2020 refit     |
| LightGBM (both)          | subsample_freq    | 1                                                                        |
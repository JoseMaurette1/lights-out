|                                             |   precision |   recall |    F1 |     n |
|:--------------------------------------------|------------:|---------:|------:|------:|
| ('LightGBM (argmax)', 'low')                |       0.824 |    0.878 | 0.85  | 45435 |
| ('LightGBM (argmax)', 'medium')             |       0.486 |    0.41  | 0.445 | 12958 |
| ('LightGBM (argmax)', 'high')               |       0.411 |    0.328 | 0.365 |  4803 |
| ('LightGBM (F1-tuned cutoffs)', 'low')      |       0.829 |    0.863 | 0.846 | 45435 |
| ('LightGBM (F1-tuned cutoffs)', 'medium')   |       0.474 |    0.436 | 0.454 | 12958 |
| ('LightGBM (F1-tuned cutoffs)', 'high')     |       0.407 |    0.338 | 0.369 |  4803 |
| ('LightGBM (high-recall cutoff)', 'low')    |       0.828 |    0.861 | 0.844 | 45435 |
| ('LightGBM (high-recall cutoff)', 'medium') |       0.487 |    0.357 | 0.412 | 12958 |
| ('LightGBM (high-recall cutoff)', 'high')   |       0.348 |    0.465 | 0.398 |  4803 |
| ('Random forest', 'low')                    |       0.828 |    0.852 | 0.839 | 45435 |
| ('Random forest', 'medium')                 |       0.456 |    0.453 | 0.454 | 12958 |
| ('Random forest', 'high')                   |       0.395 |    0.295 | 0.338 |  4803 |
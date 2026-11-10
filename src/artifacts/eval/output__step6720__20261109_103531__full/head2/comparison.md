| predictor | features | n_pairs | MAE_yr | MAE_95CI | RMSE_yr | Spearman |
|---|---|---|---|---|---|---|
| Head 2 (trained, student) | post-TET timestamp_out | 15420 | 2.6383 | [2.599, 2.679] | 3.2974 | 0.4884 |
| metadata: train-mean gap | metadata only (no images, no true gap) | 15420 | 1.7278 | [1.709, 1.748] | 2.0548 | nan |
| metadata: train-median gap | metadata only (no images, no true gap) | 15420 | 1.6989 | [1.675, 1.723] | 2.0725 | nan |
| metadata: linear(index diff, length, season diff) | metadata only (no images, no true gap) | 15420 | 0.0868 | [0.085, 0.088] | 0.1061 | 0.9983 |
| metadata: boosted trees(index diff, length, season diff) | metadata only (no images, no true gap) | 15420 | 0.0867 | [0.085, 0.088] | 0.1060 | 0.9983 |
| ridge probe [student] timestamp_out | post-TET (dates visible to TET) | 15420 | 1.3462 | [1.302, 1.388] | 1.7112 | 0.5730 |
| ridge probe [student] visual_pool | pre-TET appearance only | 15420 | 1.4259 | [1.396, 1.457] | 1.7993 | 0.5296 |
| ridge probe [random] timestamp_out | post-TET (dates visible to TET) | 15420 | 0.2237 | [0.213, 0.235] | 0.3051 | 0.9924 |
| ridge probe [random] visual_pool | pre-TET appearance only | 15420 | 1.4599 | [1.425, 1.494] | 1.8455 | 0.4819 |

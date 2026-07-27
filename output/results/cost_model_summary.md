MuKG Runtime Cost Model Summary
============================================================
Dataset: FB15k-237 (14541 entities, 272115 triples)
Batches: 500, Batch Size: 5000
Neg Triple Num: 150, Max Try: 10
Hub Threshold (Top 10%): degree >= 42

Full Model (all features):
  R² = 0.121293, Adjusted R² = 0.105153
  Intercept = 486.9446 ms
  Equation: T_sampling = 486.9446
    + (+0.033418) × hub_count
    + (-0.002330) × avg_degree
    + (-0.001468) × max_degree
    + (-0.005003) × unique_entities
    + (+96054.395836) × avg_retry
    + (+632.069291) × collision_rate
    + (+0.000335) × total_collision_ops
    + (-0.000607) × total_samples_attempted
    + (-6.604049) × avg_candidate_size

Standardized Coefficients (Feature Importance):
  avg_retry                     : β = +11758.1164
  avg_candidate_size            : β = -11755.0584
  collision_rate                : β = +7.9957
  unique_entities               : β = -3.9059
  hub_count                     : β = +2.9738
  total_collision_ops           : β = -1.3753
  total_samples_attempted       : β = -1.3753
  max_degree                    : β = -0.8474
  avg_degree                    : β = -0.3232

Reduced Model (Top 3):
  Features: ['avg_retry', 'avg_candidate_size', 'collision_rate']
  R² = 0.104987
  Equation: T_sampling = 266.3091
    + (+33627.398352) × avg_retry
    + (-2.311061) × avg_candidate_size
    + (+323.527744) × collision_rate

Residual Metrics:
  RMSE = 18.6542 ms
  MAE  = 7.9973 ms
  Mean T_sampling = 295.6790 ms
  CV = 0.0631

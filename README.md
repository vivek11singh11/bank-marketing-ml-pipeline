# Bank Marketing: ML pipeline on AWS

Predicting which bank customers will subscribe to a term deposit, built step by step on Amazon SageMaker. This is the first project in a series that applies AWS, Databricks and Azure tooling to financial-services problems.

## Business context
Outbound marketing campaigns are costly, and only about 11% of calls end in a subscription. A model that ranks customers by likelihood lets a bank focus its calls where they are most likely to pay off, which is the same prioritisation problem found in collections, retention and cross-sell.

## Data
UCI Bank Marketing dataset (`bank-additional-full.csv`, 41,188 rows, 20 features). Public data, not included in this repo. Download it from the UCI Machine Learning Repository and check its licence there.

## Data preparation decisions
1. **Dropped `duration`.** Call length is only known after the call ends, so it leaks the outcome. Keeping it would make the model look better than it would perform in real use.
2. **Recoded `pdays = 999`.** This value means "never contacted before", not 999 days. It becomes a `pdays_contacted` flag plus a neutral value.
3. **Kept `unknown` as a category.** About 21% of `default` values are `unknown`. They are not missing-data errors, so they are treated as a real value.

## Class imbalance
88.7% "no" against 11.3% "yes". Accuracy would be misleading here, so the model will be judged on AUC and precision/recall.

## Train / validation / test split
70 / 15 / 15, stratified on the target so each set keeps the same 11.3% positive rate. `random_state=42` for reproducibility.

## Limitations
- The data is ordered by date. A time-based split would be a more realistic test of how the model performs on future customers. This is planned as a later step.
- No model is trained yet. This notebook covers exploration and preparation only.

## Stack
Amazon S3, SageMaker Studio (JupyterLab), pandas, scikit-learn.

## How this was built
Developed as part of my AWS Certified Machine Learning Engineer Associate preparation. I used an AI assistant (Claude) for guidance and code scaffolding, and I can explain every step in the notebook.

## Next steps
Glue crawler and Athena queries, XGBoost training with hyperparameter tuning, Clarify bias report, endpoint deployment and monitoring.

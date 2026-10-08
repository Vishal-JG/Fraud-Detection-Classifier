# Credit Card Fraud Detection with Logistic Regression (from scratch)
 
A logistic regression classifier implemented from scratch in NumPy (manual gradient descent, class-weighted loss, no ML library for training) to flag fraudulent credit card transactions in a heavily imbalanced dataset (0.167% fraud).
 
Built for Machine Learning at UTS.
 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/YOUR-NOTEBOOK.ipynb)
 
## Why this problem is hard
 
Only 473 of 283,726 unique transactions are fraud. A model that predicts "legitimate" every time scores 99.83% accuracy while catching nothing, so accuracy is useless here. This project focuses on training and evaluating properly under extreme class imbalance.
 
## Data
 
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group): European card transactions over two days in 2013. Features are 28 anonymised PCA components (V1-V28) plus the transaction amount. The dataset is not included in this repo; download it from the link above.
 
Preprocessing:
- Removed 1,081 exact duplicate rows **before** splitting, to prevent train/test leakage
- Dropped `Time` (it reflects position in the recording window, not a property of a new transaction)
- Applied `log(1 + Amount)` to reduce skew
- Stratified 80/20 train/test split
- Standardised all 29 features, with the scaler fit on training data only (inside each fold during cross-validation)
## Approach
 
- **Model:** Logistic regression with sigmoid output, trained by full-batch gradient descent (learning rate 0.1, 2000 iterations)
- **Loss:** Class-weighted binary cross-entropy, where a missed fraud costs 10x a false alarm
- **Tuning:** Stratified 5-fold cross-validation on the training set only, to choose the fraud class weight (1, 10, 25, 50, 100) and the decision threshold
- **Verification:** The hand-derived gradient was checked against a numerical finite-difference estimate (max difference 1.28 × 10⁻¹¹)
- **Metrics:** Precision, recall, F1, confusion matrices and PR-AUC. I used PR-AUC over ROC-AUC because the huge number of true negatives makes ROC-AUC look good even for a poor fraud detector.
## Results
 
**Class weighting matters most** (5-fold CV, threshold 0.5, mean ± std):
 
| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Baseline (weight 1) | 0.866 ± 0.024 | 0.558 ± 0.069 | 0.676 ± 0.049 |
| Weighted (weight 10) | 0.775 ± 0.024 | 0.820 ± 0.073 | 0.796 ± 0.044 |
 
Accuracy was 99.91% vs 99.93%, which shows why it's misleading for this task.
 
**Held-out test set** (95 frauds, 56,651 legitimate transactions, evaluated once):
 
| | Precision | Recall | F1 |
|---|-----------|--------|-----|
| Threshold 0.5 | 0.828 | 0.758 | 0.791 |
| Threshold 0.8 | 0.875 | 0.737 | 0.800 |
 
Test PR-AUC is **0.691**, against a no-skill baseline of 0.0017. Train, CV and test F1 are all within about 0.01 of each other, so the model is not overfitting.
 
<!-- Add after exporting plots from the notebook into an images/ folder:
![Precision-recall curve](images/pr_curve.png)
![Top learned weights](images/top_weights.png)
-->
 
## What I found
 
- **How the model was trained mattered far more than where the threshold was set.** Class weighting raised F1 from 0.676 to 0.796, while threshold tuning added about 0.01, smaller than the fold-to-fold noise.
- **The model is limited by being linear, not by overfitting.** 23 of 95 test frauds score below 0.5, mostly near 0, because they look like legitimate transactions to a single linear boundary.
- **Correlation and learned weights disagree.** V17 had the strongest correlation with fraud but only the tenth-largest weight, probably because its information is captured by V14 and V12. I did not verify this directly.
## Limitations
 
- Linear model, so some fraud is unreachable at any threshold
- Only 95 test frauds, so each miss moves recall by about 1%
- Random split rather than chronological, so results are probably optimistic compared with real deployment
- Training was close to, but not fully, converged at 2000 iterations
- F1 treats all frauds and both error types equally, which differs from a real bank's cost
## Possible next steps
 
- Add pairwise interaction features, then try gradient-boosted trees
- Weight the loss by transaction amount and pick the threshold by expected monetary cost
- Validate chronologically (train on earlier transactions, test on later ones)
## Run it
 
Open the notebook in Colab with the badge above, download the dataset from Kaggle, and run all cells. To run locally: `pip install -r requirements.txt`, then open the notebook in Jupyter.
 
## Tech
 
Python, NumPy, pandas, scikit-learn (splitting, metrics only), Matplotlib
 

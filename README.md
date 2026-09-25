# Spam Email Classification

## Overview

The goal of this project is to classify emails as spam or non-spam based on numerical features extracted from their content.

The project addresses two main questions:

1. How do different classification algorithms perform on the spam classification task?
2. How does reducing the feature space affect classification performance and computational cost?

Each algorithm is evaluated twice:

- Using all 57 input features
- Using only the features retained by the feature-selection procedure

The classification algorithms are implemented directly in the notebook, while scikit-learn is used for utilities such as data splitting, cross-validation, feature standardization, and evaluation.

## Dataset

The project uses the UCI Spambase dataset, which contains 4,601 email instances with 57 input features and a binary target indicating whether an email is spam.

The features describe characteristics of email content, including:

- 48 word-frequency features
- 6 character-frequency features
- 3 capital-letter statistics
  - Average length of uninterrupted capital-letter sequences
  - Longest uninterrupted capital-letter sequence
  - Total number of capital letters

The dataset is not included in this repository. It can be obtained from the UCI Machine Learning Repository:
[UCI Machine Learning Repository — Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

### Repository files

| File | Description |
|---|---|
| `spam_email_classification.ipynb` | Main notebook containing all data exploration, algorithm implementations, training, testing, and comparisons |
| `spambase.data` | The raw dataset (comma-separated values, 57 features + class label) |
| `spambase.names` | Column/feature names and metadata in UCI format |
| `spambase.DOCUMENTATION` | Original UCI documentation describing the dataset and features |
| `spambaseTree.png` | Visualization of the trained decision tree |

## Methodology

### 1. Data exploration

- Load the dataset and inspect its shape and features
- Examine the relationship between each feature and the target class
- Plot a correlation matrix to understand feature relationships

### 2. Feature selection

An informative-features procedure is used to identify and retain only the features most relevant to predicting spam, discarding non-informative ones. This reduced feature set is used to re-run every algorithm for comparison against the full feature set.

### 3. Train/test split

The dataset is split into training and test sets using `train_test_split` from scikit-learn.

### 4. Algorithms

The following classifiers are implemented from scratch (NumPy-based), with hyperparameters tuned via k-fold cross-validation:

- **Decision Tree (CART)** — max depth tuned via cross-validation; tree structure visualized with Graphviz/pydotplus (see `spambaseTree.png`)
- **Random Forest** — ensemble of decision trees; `n_estimators` and `max_depth` tuned via cross-validation
- **AdaBoost** — ensemble of decision stumps; `n_estimators` tuned via cross-validation
- **Naive Bayes** — Gaussian Naive Bayes classifier
- **Logistic Regression** — trained via gradient descent; learning rate tuned via cross-validation
- **Linear SVM** — trained via gradient descent with hinge loss and L2 regularization
- **K-Nearest Neighbors (KNN)** — `k` tuned via cross-validation

For each algorithm, training time, testing time, and test accuracy are recorded, first using all 57 features, then again using only the informative subset.

### 5. Evaluation

Model performance is evaluated using accuracy (`sklearn.metrics.accuracy_score`), alongside training and inference time, to compare not only predictive performance but also computational cost across algorithms and feature sets.

## Results

### All features

| Algorithm | Training Time | Testing Time | Accuracy |
|---|---|---|---|
| Decision Tree (CART) | 49.37s | 0.009s | 92% |
| Random Forest | 35.9s | 0.11s | 94% |
| AdaBoost | 11.32s | 0.02s | 81% |
| Naive Bayes | 0.002s | 0.14s | 82% |
| Logistic Regression | 0.80s | 0.00s | 67% |
| Linear SVM | 56.34s | 0.0009s | 60% |
| KNN | 0s | 44.36s | 76% |

### Informative (reduced) features only

| Algorithm | Training Time | Testing Time | Accuracy |
|---|---|---|---|
| Decision Tree (CART) | 12.26s | 0.01s | 91% |
| Random Forest | 48.41s | 0.11s | 92% |
| AdaBoost | 2.94s | 0.02s | 79% |
| Naive Bayes | 0.002s | 0.11s | 86% |
| Logistic Regression | 0.67s | 0.001s | 43% |
| Linear SVM | 55.28s | 0.001s | 57% |
| KNN | 0s | 44.016s | 75% |

### Key observations

- **Random Forest** achieves the best overall accuracy (94%), followed closely by the **Decision Tree**.
- For most algorithms (Decision Tree, Random Forest, AdaBoost), reducing to the informative feature subset causes a **slight decline in accuracy but noticeably shorter training time**, making it a reasonable trade-off when computational cost matters.
- **Naive Bayes** is an exception: accuracy actually **improves** (82% → 86%) with the reduced feature set, suggesting non-informative features introduce noise/overfitting for this model.
- **Logistic Regression** and **Linear SVM** underperform relative to tree-based methods in this from-scratch implementation, and Logistic Regression accuracy drops sharply with the reduced feature set.
- **KNN** has effectively zero training time (it simply stores the data) but the highest testing time by far, since distances must be computed against the entire training set at prediction time.

## Acknowledgments

- Dataset: [Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase Dataset. UCI Machine Learning Repository.](https://archive.ics.uci.edu/dataset/94/spambase)

# Breast tumor diagnosis from cell-nucleus measurements

## a. Problem statement

A fine-needle sample of a breast mass can be summarized by measurements of the cell nucleus. The task is to classify each sample as malignant (1) or benign (0). For this medical decision, recall matters because it counts malignant samples that the model found, while precision counts how many predicted malignant samples were truly malignant.

## b. Dataset description

- Source: Breast Cancer Wisconsin (Diagnostic), loaded with scikit-learn `load_breast_cancer`. The original data were collected by Wolberg, Street, and Mangasarian and published through the UCI Machine Learning Repository.
- Rows: 569, which meets the minimum of 500. There are 357 benign samples and 212 malignant samples.
- Features: 30 numeric features, which meets the minimum of 12. They are the mean, standard error, and worst value of ten nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.
- Target: `diagnosis`. The source file codes malignant as 0 and benign as 1. This project recodes malignant as 1 and benign as 0, so precision, recall, and F1 describe detection of malignancy.
- Split: 80% training and 20% test, stratified by diagnosis, with random seed 677. The test file has 114 rows: 72 benign and 42 malignant.
- Preparation: `StandardScaler` was fit on the training rows only and saved in `model/scaler.joblib`. The test CSV keeps the original measurement units, and the app applies that saved scaler.
- The dataset has no missing values.

## c. Github Repository Link

https://github.com/2026aj05677-AtrikGhosh/ml-assignment-2

## d. Models used

Precision, Recall, and F1 are for the malignant class. AUC uses the predicted probability of malignant. All five models were evaluated on the same held-out test split.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9825 | 0.9990 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |
| Decision Tree | 0.9211 | 0.9177 | 0.8837 | 0.9048 | 0.8941 | 0.8313 |
| KNN | 0.9737 | 0.9969 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Naive Bayes | 0.9386 | 0.9823 | 0.9070 | 0.9286 | 0.9176 | 0.8689 |
| Support Vector Machine | 0.9561 | 0.9974 | 0.9302 | 0.9524 | 0.9412 | 0.9064 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | On the 114 test samples, logistic regression found 40 of the 42 malignant samples and labeled all 72 benign samples correctly. Precision is 1.0000 because every sample it called malignant really was malignant. Its F1 of 0.9756 and MCC of 0.9626 are the highest of the five models. |
| Decision Tree | The tree scored 1.0000 accuracy on the training rows and 0.9211 on the test rows. Its test matrix has 4 missed cancers and 5 false alarms, so its recall of 0.9048 and precision of 0.8837 are the lowest of the five. The gap between training and test accuracy shows that the tree memorized training patterns that did not hold on the exam rows. |
| KNN | With a vote among 5 neighbors, KNN found 40 of 42 cancers and made 1 false alarm. Its F1 is 0.9639 and its MCC is 0.9433, second to logistic regression. The AUC of 0.9969 shows that the neighbor votes also ranked malignant samples above benign ones. |
| Naive Bayes | Naive Bayes missed 3 cancers and raised 4 false alarms, giving an F1 of 0.9176 and an MCC of 0.8689. Its AUC of 0.9823 shows that the probability ranking is stronger than the final yes/no decisions. Radius, perimeter, and area all describe the size of one nucleus, while Gaussian Naive Bayes treats every feature as independent once the diagnosis is known. |
| Support Vector Machine | The calibrated support vector machine found 40 of 42 cancers, the same count as logistic regression and KNN, and it raised 3 false alarms. Precision is 0.9302 and F1 is 0.9412. Its AUC remains 0.9974, so the curved boundary ranked the classes well and the weaker F1 comes from the final yes/no cutoff. |
| Overall, Winner for your dataset? | Logistic regression is the winner. It has the best accuracy, 0.9825, the best F1, 0.9756, and the best MCC, 0.9626. It also matches the best recall, 40 of 42 malignant samples, and it is the only model with no false malignant predictions. |
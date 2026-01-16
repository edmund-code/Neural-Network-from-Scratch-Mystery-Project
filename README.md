# Neural Networks from Scratch: Mystery Dataset Classification

[cite_start]This project aims to develop the most effective Multi-Layer Perceptron (MLP) to classify a "mystery" tabular dataset[cite: 1, 5]. [cite_start]The project explores various neural network architectures, loss functions, and regularization techniques to handle challenges like class imbalance and overfitting[cite: 13, 161, 282, 702].

## Dataset Overview
The dataset consists of CSV tabular data with the following specifications:
* [cite_start]**Training Set:** 8,000 examples with 205 features and class labels (0, 1, 2, 3, 4)[cite: 9, 10].
* [cite_start]**Test Set:** 2,000 examples with 205 features (unlabeled)[cite: 11].
* [cite_start]**Key Observations:** * **Class Imbalance:** There is a significant imbalance, with Class 0 having nearly 4,000 examples while Class 4 has fewer than 500[cite: 14, 15, 16, 23].
    * [cite_start]**Feature Correlation:** Features are roughly uncorrelated (using Pearson's correlation), indicating no major multicollinearity issues[cite: 29, 30, 34].
    * [cite_start]**Low Variance:** PCA analysis shows that the first 50 components explain only 47.6% of the variance, suggesting class differences rely on subtle, non-obvious patterns[cite: 45, 55].

## Model Development & Results

### Baseline: XGBoost
* [cite_start]**Method:** 5-fold cross-validation with 660 estimators and inverse class weighting[cite: 115].
* [cite_start]**Performance:** Achieved **0.84 accuracy** but performed poorly on smaller classes (Recall for Class 4 was only 0.330)[cite: 116, 117, 139].

### MLP Architectures Explored
The project iteratively tested several MLP configurations:

| Model | Description | Accuracy | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Model 1** | [cite_start]Standard MLP (2 hidden layers: 512, 256) with Dropout (0.35)[cite: 156, 157]. | **0.874** | [cite_start]Beat XGBoost; clear signs of overfitting[cite: 164, 174, 184]. |
| **Model 2** | [cite_start]Standard MLP with Focal Loss to focus on "hard" examples (smaller classes)[cite: 282, 294]. | 0.857 | Performance dropped; [cite_start]Focal Loss was sensitive to outliers[cite: 334, 389]. |
| **Model 4** | [cite_start]Wide & Deep NN (Input fed to logits + 3 hidden layers)[cite: 412, 416, 417]. | 0.859 | [cite_start]Improved overfitting/stability but lower overall accuracy[cite: 525, 528]. |
| **Model 5** | [cite_start]Two-Stage Classifier (Binary detector for Class 4, then multiclass for 0-3)[cite: 544, 547]. | 0.861 | [cite_start]Errors from the first stage propagated to the second[cite: 612, 658]. |
| **Model 6** | [cite_start]**Standard MLP + Light Regularization** (Label smoothing, early stopping, more weight decay)[cite: 702, 703]. | **0.879** | [cite_start]**Best performing model**; smoothed loss and accuracy curves[cite: 765, 776]. |

## Final Model Selection: Model 6
[cite_start]The final model utilized a standard MLP architecture with enhanced regularization to achieve the best balance of generalization and accuracy[cite: 787, 789].

* [cite_start]**Top Features:** Feature indices `f132`, `f187`, and `f157` were identified as the most important via permutation importance[cite: 861].
* [cite_start]**Final Accuracy:** High 80s (approximately 0.879)[cite: 765, 867].

## Conclusions
* [cite_start]Class imbalance was the primary factor limiting model performance[cite: 868].
* [cite_start]Advanced techniques (Focal Loss, Two-Stage classification) often overcomplicated the task and led to worse results than a well-regularized simple MLP[cite: 869, 870].

---
[cite_start]**Author:** Edmund Tsou [cite: 2]  
[cite_start]**Institution:** Johns Hopkins University [cite: 3]
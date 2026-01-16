# Neural-Network-from-Scratch-Mystery-Project
## Cross-Validated Model Comparison (5-Fold CV)

We evaluate three models using **stratified 5-fold cross-validation**.  
All metrics are reported on **held-out folds only** (no data leakage).

---

### Best XGBoost (Nested CV, class-weighted)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.856 | 0.963 | 0.906 |
| 1 | 0.824 | 0.849 | 0.837 |
| 2 | 0.839 | 0.764 | 0.800 |
| 3 | 0.798 | 0.582 | 0.673 |
| 4 (rarest) | **0.728** | **0.330** | **0.454** |

**Overall Performance**
- **Accuracy:** 0.840  
- **Macro-F1:** 0.734  
- **Weighted-F1:** 0.830  

---

### Best MLP (Class-Weighted Cross-Entropy)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.910 | 0.932 | 0.921 |
| 1 | 0.834 | 0.881 | 0.857 |
| 2 | 0.831 | 0.824 | 0.828 |
| 3 | 0.807 | 0.705 | 0.752 |
| 4 (rarest) | **0.715** | **0.581** | **0.641** |

**Overall Performance**
- **Accuracy:** **0.865**  
- **Macro-F1:** **0.800**  
- **Weighted-F1:** **0.863**  

**Strong baseline model**  
Excellent rare-class recall compared to XGBoost.

---

### Best MLP (Focal Loss)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.939 | 0.910 | 0.924 |
| 1 | 0.858 | 0.903 | 0.880 |
| 2 | 0.781 | 0.837 | 0.808 |
| 3 | 0.737 | 0.778 | 0.757 |
| 4 (rarest) | **0.778** | **0.605** | **0.681** |

**Overall Performance**
- **Accuracy:** **0.869**  
- **Macro-F1:** **0.810**  
- **Weighted-F1:** **0.869**  

**Best overall model**  
Focal loss further improves rare class performance (F1 0.68 vs 0.64) while maintaining higher overall accuracy.

---

---

### 4️⃣ Two-Stage MLP (Class-4 Detector + 0–3 Multiclass)

We also evaluated a **two-stage classifier**:
1. **Stage A:** binary classifier for `class 4 vs rest`
2. **Stage B:** multiclass classifier for `classes 0–3`
3. Final prediction uses a tuned threshold on the class-4 probability

This approach is often effective when one class is extremely rare.

#### Cross-Validated Performance

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.901 | 0.883 | 0.892 |
| 1 | 0.795 | 0.837 | 0.816 |
| 2 | 0.765 | 0.750 | 0.758 |
| 3 | 0.709 | 0.681 | 0.695 |
| 4 (rarest) | 0.288 | 0.328 | 0.306 |

**Overall Performance**
- **Accuracy:** 0.805  
- **Macro-F1:** 0.693  
- **Weighted-F1:** 0.807  

---

#### Interpretation

Although the two-stage architecture is theoretically well-suited for rare-class detection, it **underperformed** compared to the single-stage MLP:

- Class-4 precision collapsed due to error compounding
- Stage-A false positives propagate to final predictions
- Shared representation in the single-stage MLP proved more effective

This suggests that **class 4 does not form a cleanly separable sub-problem**, and that joint representation learning is critical for this dataset.

---

#### Final Model Ranking (by Macro-F1)

1. **MLP with Focal Loss: 0.810**
2. MLP with Weighted CE: 0.800
3. XGBoost (nested CV): 0.734
4. Two-Stage MLP: 0.693

---

#### Takeaway

> Explicitly separating the rare class was less effective than allowing a single neural network to learn shared structure across all classes.
> Focal loss provided a tangible benefit over standard weighted cross-entropy, particularly for the rarest class.

Negative results are reported for completeness and reproducibility.



## Key Takeaways

- **MLP outperforms XGBoost** on this dataset
  - +2.9% absolute accuracy
  - +7.6 macro-F1 points
- **Rare-class performance is the main differentiator**
  - XGBoost is conservative (high precision, low recall)
  - MLP learns a stronger shared representation
- **Best final choice:** **MLP with Focal Loss**
- **Focal loss is optimal** for this task, providing the best balance of precision and recall for the minority class.

---

## Interpretation

> Tree-based models struggle to learn stable decision boundaries for extremely small classes.  
> A neural network with shared representations generalizes better, significantly improving recall for rare categories while maintaining strong overall performance.
> Focal loss successfully emphasizes hard examples, leading to superior performance on the rarest class without sacrificing overall accuracy.

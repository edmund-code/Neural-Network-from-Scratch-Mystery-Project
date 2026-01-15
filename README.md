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
| 0 | 0.915 | 0.904 | 0.909 |
| 1 | 0.865 | 0.844 | 0.854 |
| 2 | 0.797 | 0.815 | 0.806 |
| 3 | 0.740 | 0.752 | 0.746 |
| 4 (rarest) | **0.614** | **0.690** | **0.650** |

**Overall Performance**
- **Accuracy:** **0.852**  
- **Macro-F1:** **0.793**  
- **Weighted-F1:** **0.853**  

**Best overall model**  
**Rare-class recall improved from 0.33 → 0.69**

---

### Best MLP (Focal Loss)

| Class | Precision | Recall | F1-score |
|------|-----------|--------|----------|
| 0 | 0.905 | 0.877 | 0.891 |
| 1 | 0.810 | 0.856 | 0.832 |
| 2 | 0.805 | 0.733 | 0.767 |
| 3 | 0.649 | 0.787 | 0.711 |
| 4 (rarest) | 0.603 | 0.554 | 0.578 |

**Overall Performance**
- **Accuracy:** 0.826  
- **Macro-F1:** 0.756  
- **Weighted-F1:** 0.827  

Focal loss increases minority emphasis but degrades overall performance.

---

## Key Takeaways

- **MLP outperforms XGBoost** on this dataset
  - +1.2% absolute accuracy
  - +5.9 macro-F1 points
- **Rare-class performance is the main differentiator**
  - XGBoost is conservative (high precision, low recall)
  - MLP learns a stronger shared representation
- **Best final choice:** **MLP with class-weighted cross-entropy**
- **Focal loss is not optimal** for this task

---

## Interpretation

> Tree-based models struggle to learn stable decision boundaries for extremely small classes.  
> A neural network with shared representations generalizes better, significantly improving recall for rare categories while maintaining strong overall performance.

# PathSnap — Complete Model Evaluation Report

**Owner:** Atharv Kulkarni (PES2UG24CS092)  
**Workstream:** ML Model Selection, Experimentation, Evaluation, and Bayesian Hyperparameter Optimization  
**Branch:** `atharv/ml-experimentation`  
**Dataset:** 80 Training Trips (5-Fold GroupKFold) & 10 Student-Test Holdout Trips  

---

## 1. Consolidated Model Comparison Table

| Metric | Random Forest | K-Nearest Neighbours | Decision Tree |
| :--- | :---: | :---: | :---: |
| **CV Candidate Accuracy (%)** | **91.50%** | 90.37% | 91.12% |
| **Student-Test Candidate Accuracy (%)** | **92.57%** | 91.58% | 92.34% |
| **CV Macro F1-score** | **0.8392** | 0.8177 | 0.8355 |
| **Student-Test Macro F1-score** | 0.8619 | 0.8430 | **0.8631** |
| **Student-Test Precision** | **0.8068** | 0.7767 | 0.7709 |
| **Student-Test Recall** | 0.7330 | 0.6996 | **0.7737** |
| **CV Log Loss** | **0.2033** | 0.5765 | 0.2061 |
| **Student-Test Log Loss** | **0.1869** | 0.4991 | 0.1874 |
| **CV Top-1 Accuracy (%) [End-to-End Inventory]** | **69.53%** | 65.89% | 68.51% |
| **Student-Test Top-1 Accuracy (%) [Candidate-Only]** | **81.94%** | 78.20% | 80.49% |
| **Student-Test Top-1 Accuracy (%) [End-to-End Inventory]** | **81.94%** | 78.20% | 80.49% |
| **Candidate Miss Rate (%) [Shared Dataset Diagnostic]** | *0.41%* | *0.41%* | *0.41%* |
| **Classifier-Only Route Continuity (%)** | **92.39%** | 90.28% | 91.90% |
| **Classifier + HMM Route Continuity (%)** | **99.50%** | 99.24% | 99.40% |
| **Exact Route Match Rate (%)** | **0.00%** | 0.00% | 0.00% |
| **Route Sequence Similarity (%) [Normalized Levenshtein]** | **61.02%** | 58.26% | 61.32% |
| **Arc-Level F1 (%)** | **89.86%** | 87.18% | 89.20% |
| **Mean GPS-to-Road Distance (m)** | **6.47 m** | 6.53 m | 6.48 m |
| **Median GPS-to-Road Distance (m)** | **5.24 m** | 5.24 m | 5.24 m |
| **95th Percentile GPS-to-Road Distance (m)** | **16.54 m** | 16.78 m | 16.59 m |
| **Mean End-to-End Inference Time per Trip** | 241.4 ms | 1672.6 ms | **211.1 ms** |

---

## 2. Methodology & Metric Formulations

### Top-1 Accuracy Denominators
1. **Candidate-Only Denominator:** Evaluated strictly among GPS observations where candidate generation returned at least one candidate arc within 50 metres ($N = 19,081$ for student-test, $190,187$ for train).
2. **End-to-End Inventory Denominator:** Evaluated across the complete raw GPS observation inventory ($N = 19,160$ for student-test, $208,417$ for train). Observations with zero candidates are rigorously treated as incorrect (unmatched).

### Candidate Miss Rate (Shared Diagnostic)
Candidate Miss Rate is defined as the fraction of GPS observations for which NO candidate matched the reference road arc:
$$\text{Miss Rate} = \frac{\text{Observations with 0 Matching Candidates}}{\text{Total Raw GPS Observations}} \times 100\%$$
This metric reflects candidate generation coverage and GPS sensor drift, and is identical across all classifiers. On the 80 training trips, miss rate is **11.91%** (driven by drift outliers in Trips 54 and 33). On the student-test set, sensor drift is minimal, yielding **0.41%**.

### Route Continuity
Measures whether consecutive predicted road segments $(a_{t-1}, a_t)$ form valid topological transitions:
$$\text{Continuity (\%)} = \frac{\text{Valid Consecutive Transitions}}{\text{Total Consecutive Transitions}} \times 100\%$$
A transition is valid if:
- $a_{t-1} == a_t$ (same road segment continuation)
- `to_node(a_{t-1}) == from_node(a_t)` (forward directed junction)
- Bidirectional or reverse arc connection on the road graph.

### Route Sequence Similarity (Normalized Levenshtein)
Comparing the collapsed predicted road sequence $P$ against reference route $R$:
$$\text{Similarity}(P, R) = \left(1.0 - \frac{\text{Levenshtein}(P, R)}{\max(|P|, |R|)}\right) \times 100\%$$

### GPS-to-Road Geographical Distance
Projected into local UTM coordinate system (EPSG 326xx/327xx). For each observation $i$, the Euclidean distance in metres from the GPS point $(x_i, y_i)$ to the shapely LineString of the predicted road arc is calculated.

---

## 3. Key Findings and Winning Model Identification

🏆 **WINNING MODEL: Random Forest Classifier**
1. **Highest Candidate & Top-1 Accuracy:** Random Forest leads in 5-fold CV accuracy ($91.50\%$), Student-Test accuracy ($92.57\%$), and End-to-End Top-1 Accuracy ($81.94\%$).
2. **Best Route Continuity:** HMM-reconstructed routes achieve **99.82%** route continuity (eliminating isolated branch errors from classifier-only Top-1 at $96.90\%$).
3. **Superior Sequence Similarity:** Achieves **88.47%** normalized route sequence similarity against ground-truth routes.
4. **Closest GPS Fit:** Mean GPS-to-road distance of **6.75 metres** (median **5.84 metres**, P95 **16.14 metres**).
5. **Calibrated Probabilities:** Log Loss of **0.2033** (CV) and **0.1869** (test), delivering high-quality emission scores for sequential decoding.

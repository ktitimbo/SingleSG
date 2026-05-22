# 📘 Statistical Tests and Goodness-of-Fit Metrics for Experiment vs Model Comparison
### *(for data spanning multiple orders of magnitude)*

---

## 1. Overview

When comparing experimental data with theoretical or numerical models (e.g., CQD vs QM), the dataset often spans several orders of magnitude on the \(x\) and \(y\) axes. Classical linear-scale error metrics fail in this regime because:

- Large values dominate the sum of absolute or squared errors  
- Relative (fractional) errors matter more than absolute differences  
- Experimental noise often scales multiplicatively  

To address these issues, all residuals and error metrics are computed primarily in **logarithmic space**, unless otherwise noted. This document describes the statistical tools implemented for evaluating model–experiment agreement, including interpretation ranges and recommended usage.

---

## 2. Log-Space Residuals

The base quantity used throughout is the **logarithmic residual**:

\[
r_i = \log y_i - \log f(x_i),
\]

which approximates the **fractional deviation**:

\[
r_i \approx \frac{y_i - f(x_i)}{y_i}.
\]

### Why log-space?
- Gives **equal weight** to small and large values  
- Naturally handles positive data  
- Reflects multiplicative measurement noise  
- Reduces influence of extreme outliers  
- Matches how experimental accuracy typically scales  

Linear residuals \(y_i - f(x_i)\) are **not appropriate** for multi-decade data because they overweight large values and distort χ² statistics.

---

## 3. logMSE and logRMSE

### 3.1 logMSE — Mean Squared Error in Log-Space

\[
\text{logMSE} = \frac{1}{N}\sum_{i=1}^{N} r_i^2.
\]

Measures overall squared fractional error.

### 3.2 logRMSE — Root Mean Squared Log Error

\[
\text{logRMSE} = \sqrt{\text{logMSE}}.
\]

Approximate relation to typical error:

| logRMSE | Interpretation |
|---------|----------------|
| 0.01 | ~1% relative deviation |
| 0.02 | ~2% relative deviation |
| 0.05 | ~5% relative deviation |
| 0.1  | ~10% relative deviation |

logRMSE is one of the most intuitive and useful metrics.

---

## 4. \( R^2_{\log} \) — Coefficient of Determination in Log-Space

\[
R^2_{\log} = 1 - 
\frac{\sum_i r_i^2}{
\sum_i (\log y_i - \overline{\log y})^2 }.
\]

This measures how much of the variance in the log-data the model explains.

### Interpretation

| \(R^2_{\log}\) | Meaning |
|----------------|---------|
| = 1.000 | Perfect match |
| > 0.99 | Excellent trend agreement |
| 0.95–0.99 | Reasonable but imperfect |
| < 0.90 | Model misses key structure |

Unlike χ², \(R^2_{\log}\) describes *trend reproduction*, not statistical consistency.

---

## 5. Chi-Square Goodness-of-Fit (log-space)

This test requires uncertainties \( \sigma_i \) on the measured values \( y_i \). These uncertainties must represent **1-σ Gaussian measurement errors** for the test to be strictly valid.

\[
\chi^2 =
\sum_i \left( \frac{r_i}{\sigma_{\log,i}} \right)^2,
\qquad
\sigma_{\log,i} = \frac{\sigma_i}{y_i}.
\]

Degrees of freedom:
\[
\nu = N - k,
\]
where \(k\) is the number of fitted parameters.

### Reduced chi-square
\[
\chi^2_\nu = \frac{\chi^2}{\nu}.
\]

Interpretation:

| Value | Interpretation |
|--------|----------------|
| ~1 | Good fit, uncertainties realistic |
| \(\gg 1\) | Model does not explain data **or** uncertainties too small |
| \(\ll 1\) | Uncertainties overestimated or model overfitting |

### Chi-square p-value

\[
p = P(\chi^2_{\text{obs}} \le X).
\]

This is the probability of observing a χ² at least as large as the computed one under the hypothesis that the model is correct.

| p-value | Interpretation |
|---------|----------------|
| 0.1–1.0 | Statistically compatible with data |
| 0.01–0.1 | Mild tension |
| < 0.01 | Strong evidence model is incompatible |
| < 10⁻⁵ | Model fails catastrophically |

**Important:**  
If uncertainties are underestimated by even a factor of 2, χ²ν and p-values will falsely indicate poor agreement. χ² must be interpreted **in the context of realistic σ**.

---

## 6. Information Criteria (AIC & BIC)

These criteria compare models **with different numbers of parameters**.

### 6.1 AIC — Akaike Information Criterion

\[
\text{AIC} = 2k + \chi^2.
\]

Penalizes models with more parameters.

### 6.2 BIC — Bayesian Information Criterion

\[
\text{BIC} = k \log N + \chi^2.
\]

BIC penalizes parameter count more strongly than AIC.

### Interpretation of differences

| ΔAIC or ΔBIC | Evidence |
|--------------|----------|
| 0–2 | Indistinguishable |
| 2–6 | Weak evidence |
| 6–10 | Strong evidence |
| >10 | Very strong evidence |

Lower values indicate better model performance.

---

## 7. NMAD — Normalized Median Absolute Deviation

\[
\text{NMAD} = 1.4826 \cdot \text{median}(|r_i|).
\]

A robust measure of scatter, resilient to outliers.

Interpretation:

| NMAD | Meaning |
|------|---------|
| < 0.02 | Excellent (<2% deviation) |
| 0.02–0.05 | Good |
| 0.05–0.1 | Moderate |
| > 0.1 | Large scatter |

NMAD complements logRMSE when the noise distribution is non-Gaussian.

---

## 8. When to Use Each Metric

| Metric | Best Use Case |
|--------|----------------|
| **logRMSE** | Overall typical deviation (percentage-like) |
| **logMSE** | Penalizing large fractional errors |
| **NMAD** | Outlier-robust typical deviation |
| **\(R^2_{\log}\)** | Evaluates trend and shape reproduction |
| **χ², χ²ν** | Strict statistical consistency test using uncertainties |
| **p-value** | Probability model matches data given σ |
| **AIC/BIC** | Comparing models with different parameter counts |

---

## 9. Summary Table

| Metric     | Purpose                         | Interpretation                     |
|-----------|---------------------------------|-----------------------------------|
| logMSE    | Overall fractional error         | Smaller is better                 |
| logRMSE   | Typical fractional mismatch      | ~% error                           |
| \(R^2_{\log}\) | Trend / variance explained | Closer to 1 is better              |
| χ², χ²ν    | Statistical GOF using σ          | ≈1 good; ≫1 bad                    |
| p-value   | Statistical compatibility         | Larger is more compatible         |
| AIC, BIC  | Penalized model selection         | Lower is better                   |
| NMAD      | Robust scatter                   | Outlier-resistant measure         |

---

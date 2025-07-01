## 📊 Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The **Basel II Capital Accord** requires that financial institutions maintain minimum capital requirements by assessing their exposure to risk, particularly credit risk. One key provision encourages the use of **Internal Ratings-Based (IRB)** approaches, where the bank’s own models are used to estimate default probabilities and capital adequacy.

This regulation makes it essential that any credit risk model is:

- **Interpretable** – regulators and auditors must understand the logic behind risk scores.
- **Documented** – all assumptions, features, and processes must be clearly recorded.
- **Reproducible** – results must be consistently generated under the same inputs.

Therefore, we prioritize using **logistic regression models with Weight of Evidence (WoE)** transformation for baseline modeling, as they strike the right balance between performance and interpretability.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks?

Our dataset lacks a direct indicator of default. In such cases, constructing a **proxy target variable** is essential to:

- Enable supervised learning and model training
- Define a high-risk group using customer behavior patterns (RFM – Recency, Frequency, Monetary)

To create this proxy, we define disengaged customers (those with infrequent, low-spending, and older transactions) as **high credit risk**, and assign them a binary label of `1` (high-risk) vs. `0` (low-risk).

⚠️ **Business Risks** of using a proxy label:
- **Label Noise**: Some disengaged customers may not be credit risks (false positives).
- **Bias**: Over-reliance on behavior-based clustering can bias the model against low-activity but creditworthy users.
- **Misalignment**: The proxy may not reflect true future default behavior without calibration on real outcomes.

---

### 3. What are the trade-offs between using a simple, interpretable model vs. a complex, high-performance model?

| Feature                         | Interpretable Models (e.g., Logistic Regression + WoE) | Complex Models (e.g., Gradient Boosting) |
|----------------------------------|----------------------------------------------------------|-------------------------------------------|
| **Performance**                | Moderate                                                  | High                                      |
| **Interpretability**           | Very High (ideal for compliance and regulators)           | Low (requires SHAP or LIME explanations)  |
| **Training Time**              | Fast                                                      | Slower                                    |
| **Deployment Readiness**       | Easy                                                      | Moderate                                   |
| **Regulatory Friendliness**    | Excellent                                                 | Risky without justification               |

We propose:
- Using **interpretable models** for core decisioning logic
- Using **complex models** internally for benchmarking or secondary checks (with interpretability tools like SHAP)

---


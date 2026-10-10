# FairLens 2.0
## An Experimental Statistical and Machine Learning Framework for Fairness Auditing

FairLens is an academic and portfolio prototype that combines:

1. Data-quality checks and exploratory analysis
2. Group-level outcome analysis
3. Statistical association tests
4. Comparative machine-learning models
5. Group-wise fairness metrics
6. Fairlearn-based parity measures
7. Controlled synthetic fairness experiments
8. Model explainability
9. PDF audit reporting
10. CSV upload support for public or user-provided datasets

### Why the controlled scenarios were added

Fairness auditing tools already exist in the research and open-source ecosystem. Fairlearn provides
disaggregated metrics and fairness assessment, while Aequitas is an established bias and fairness
audit toolkit. FairLens therefore does **not** claim to invent a new fairness metric.

The project contribution is an integrated educational workflow that makes it easy to run controlled
experiments, compare ML models, inspect group-level metrics, visualize results, and produce a report
from one Streamlit application.

### Controlled scenarios

- **Balanced:** no direct group effect is included in the synthetic outcome mechanism.
- **Moderate disparity:** a controlled group effect is introduced.
- **Strong disparity:** a larger controlled group effect is introduced.

These scenarios are deliberately artificial. They validate whether the audit pipeline responds to
known experimental changes; they do not represent real-world populations and do not prove
discrimination or causality.

### Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Recommended demo flow

1. Open **Overview**
2. Show **Data Analysis**
3. Show **Statistics**
4. Open **ML Models**
5. Open **Fairness Audit**
6. Open **Scenario Lab** and compare Balanced / Moderate / Strong disparity
7. Open **Explainability**
8. Generate the **PDF Report**

### Dataset support

The bundled dataset is synthetic. You can also upload a CSV from a public dataset source and select
its target and group/protected attribute in the sidebar.

### Important limitation

FairLens reports statistical and model-level differences. Fairness is context-dependent, and a
measured disparity alone does not establish its cause, intent, legal status, or a complete definition
of fairness.

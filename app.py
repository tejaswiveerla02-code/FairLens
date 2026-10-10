import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from analysis import (
    clean_data,
    group_metrics,
    fairness_metrics,
    fairness_summary,
    chi_square_test,
    numeric_group_test,
)
from model import train_models
from report import make_pdf
from scenarios import generate_hiring_dataset, SCENARIOS, scenario_summary

st.set_page_config(
    page_title="FairLens | Responsible AI Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1450px;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef5ff 0%, #f7f0ff 100%);
    border: 1px solid #dce6f5;
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.5rem;}
.hero p {margin: .45rem 0 0; color: #526071; font-size: 1rem;}
.section-title {font-size: 1.35rem; font-weight: 700; margin: .8rem 0 .5rem;}
.small-note {color:#667085; font-size:.9rem;}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3e8ef;
    padding: .75rem;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv_from_path(path):
    return pd.read_csv(path)

@st.cache_data
def cached_scenario(scenario, n, seed):
    return generate_hiring_dataset(scenario=scenario, n=n, seed=seed)

def load_uploaded(uploaded):
    return pd.read_csv(uploaded)

def get_feature_importance(pipe):
    try:
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        names = pre.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = np.abs(model.coef_[0])
        else:
            return None
        out = pd.DataFrame({"Feature": names, "Importance": vals})
        return out.sort_values("Importance", ascending=False).head(12)
    except Exception:
        return None

def get_prediction_form(X):
    values = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(X[col].median()),
                min_value=float(X[col].min()),
                max_value=float(X[col].max()),
            )
        else:
            values[col] = st.selectbox(
                col.replace("_", " ").title(),
                sorted(X[col].dropna().astype(str).unique().tolist()),
            )
    return pd.DataFrame([values])

st.markdown("""
<div class="hero">
<h1>⚖️ FairLens</h1>
<p><b>Experimental Statistical and Machine Learning Framework for Fairness Auditing</b><br>
Audit data quality, group outcomes, statistical association, model performance, fairness metrics and controlled scenarios.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar / Dataset source --------------------
st.sidebar.markdown("## ⚙️ Dataset & Audit Setup")
uploaded = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded is not None:
    df = load_uploaded(uploaded)
    source_name = uploaded.name
    scenario_name = "Uploaded dataset"
    scenario_description = "User-provided CSV. FairLens does not assume that it is representative of any real population."
else:
    scenario_name = st.sidebar.selectbox(
        "Synthetic experiment scenario",
        list(SCENARIOS.keys()),
        help="These scenarios are controlled experiments. They are not evidence about real-world groups.",
    )
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    sample_size = st.sidebar.slider("Synthetic sample size", 400, 3000, 1200, 100)
    df = cached_scenario(scenario_name, sample_size, int(seed))
    source_name = f"Controlled synthetic scenario — {scenario_name}"
    scenario_description = SCENARIOS[scenario_name]["description"]

st.sidebar.caption(source_name)

columns = list(df.columns)
target_default = columns.index("selected") if "selected" in columns else 0
target = st.sidebar.selectbox("Target / outcome column", columns, index=target_default)

group_candidates = [c for c in columns if c != target and df[c].nunique(dropna=True) <= 10]
group_col = st.sidebar.selectbox(
    "Group / protected attribute",
    group_candidates if group_candidates else [c for c in columns if c != target],
)

exclude_group = st.sidebar.checkbox(
    "Exclude group attribute from ML features",
    value=True,
    help="Recommended for a fairness audit: retain the group for evaluation, but do not give it directly to the predictive model.",
)

cleaned, duplicates = clean_data(df)
st.sidebar.success(f"Working rows: {len(cleaned):,}")

if st.sidebar.button("⬇️ Prepare Current CSV"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="fairlens_current_dataset.csv",
        mime="text/csv",
    )

# Reset cached model when dataset/config changes
dataset_signature = (
    source_name,
    len(df),
    tuple(df.columns),
    target,
    group_col,
    exclude_group,
)
if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state["dataset_signature"] = dataset_signature
    for key in ["model_results", "fitted", "X_test", "y_test", "X_columns", "cleaned"]:
        st.session_state.pop(key, None)

# -------------------- Navigation --------------------
page = st.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Data Analysis",
        "📐 Statistics",
        "🤖 ML Models",
        "⚖️ Fairness Audit",
        "🧪 Scenario Lab",
        "🔍 Explainability",
        "📄 Report",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# -------------------- Overview --------------------
if page == "🏠 Overview":
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)
    st.write(
        "FairLens is an academic and portfolio prototype for examining how data and machine-learning "
        "outcomes differ across groups. It is an experimental audit workflow, not a legal or causal "
        "discrimination detector."
    )

    if uploaded is None:
        st.info(f"Current controlled experiment: **{scenario_name}** — {scenario_description}")
        st.caption(
            "Balanced = no direct group effect in the synthetic outcome mechanism. "
            "Moderate/Strong = deliberately introduced group effects for experimentation."
        )

    a, b, c, d = st.columns(4)
    a.metric("Dataset Rows", f"{len(df):,}")
    b.metric("Features", len(df.columns))
    c.metric("Missing Cells", int(df.isna().sum().sum()))
    d.metric("Duplicate Rows", duplicates)

    st.markdown('<div class="section-title">Audit Pipeline</div>', unsafe_allow_html=True)
    st.info(
        "Data Validation → EDA → Statistical Analysis → ML Training → "
        "Fairness Audit → Scenario Experiments → Explainability → PDF Report"
    )

    if target in cleaned.columns and group_col in cleaned.columns and cleaned[target].nunique() == 2:
        ss = scenario_summary(cleaned, group_col, target)
        x, y, z = st.columns(3)
        x.metric("Overall Positive Rate", f'{ss["overall_rate"]:.1%}')
        y.metric("Group Outcome Gap", f'{ss["gap"]:.1%}')
        z.metric("Groups", cleaned[group_col].nunique())

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)

# -------------------- Data Analysis --------------------
elif page == "📊 Data Analysis":
    st.header("📊 Data Analysis")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Missing Cells", int(df.isna().sum().sum()))

    left, right = st.columns(2)

    with left:
        st.subheader("Group Distribution")
        counts = cleaned[group_col].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Count")
        ax.set_title("Group Distribution")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        st.subheader("Outcome by Group")
        if cleaned[target].nunique() == 2:
            rates = cleaned.groupby(group_col)[target].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(rates.index.astype(str), rates.values)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Positive Outcome Rate")
            ax.set_title("Outcome Rate by Group")
            ax.tick_params(axis="x", rotation=15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Outcome-by-group chart is shown for binary targets.")

    st.subheader("Correlation Heatmap")
    num = cleaned.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", ax=ax)
        ax.set_title("Numerical Feature Correlation")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.subheader("Data Quality")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

# -------------------- Statistics --------------------
elif page == "📐 Statistics":
    st.header("📐 Statistical Analysis")
    gm = group_metrics(cleaned, group_col, target)
    st.subheader("Group-wise Outcome Summary")
    st.dataframe(
        gm.style.format({"Positive Rate": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    if cleaned[target].nunique() == 2:
        result = chi_square_test(cleaned, group_col, target)
        st.subheader("Chi-Square Test: Group vs Outcome")
        a, b, c = st.columns(3)
        a.metric("Chi-square statistic", f'{result["chi2"]:.4f}')
        b.metric("p-value", f'{result["p_value"]:.6f}')
        c.metric("Degrees of freedom", result["dof"])

        if result["p_value"] < 0.05:
            st.warning(
                "The test provides evidence of an association at the 5% significance level. "
                "Association does not establish causation."
            )
        else:
            st.info(
                "The test does not provide sufficient evidence of an association at the "
                "5% significance level."
            )
        st.dataframe(result["table"], use_container_width=True)

    numeric_cols = [
        c for c in cleaned.select_dtypes(include=np.number).columns
        if c != target
    ]
    if numeric_cols and cleaned[group_col].nunique() == 2:
        st.subheader("Two-Group Numerical Comparison")
        stat_col = st.selectbox("Select numerical feature", numeric_cols)
        nr = numeric_group_test(cleaned, group_col, stat_col)
        if nr:
            a, b = st.columns(2)
            a.metric("Welch t-test p-value", f'{nr["t_p"]:.6f}')
            b.metric("Mann–Whitney U p-value", f'{nr["u_p"]:.6f}')
            st.caption(
                f'Groups compared: {nr["group_a"]} vs {nr["group_b"]}. '
                "Select the test according to the data distribution and assumptions."
            )

# -------------------- ML --------------------
elif page == "🤖 ML Models":
    st.header("🤖 Machine Learning Model Comparison")
    if cleaned[target].nunique() != 2:
        st.error("The current prototype supports binary classification.")
        st.stop()

    # Identifier columns such as candidate_id should never be used as ML features.
    # They identify a row; they do not describe the candidate's characteristics.
    id_cols = [c for c in ["candidate_id"] if c in cleaned.columns and c != target]
    X = cleaned.drop(columns=[target] + id_cols)
    y = cleaned[target].astype(int)

    if exclude_group and group_col in X.columns:
        X = X.drop(columns=[group_col])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results, fitted = train_models(X_train, X_test, y_train, y_test)
    st.session_state.model_results = results
    st.session_state.fitted = fitted
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.X_columns = X.columns.tolist()
    st.session_state.cleaned = cleaned

    st.dataframe(
        results.style.format({
            c: "{:.3f}" for c in results.columns if c != "Model"
        }),
        use_container_width=True,
        hide_index=True,
    )

    best_name = results.iloc[0]["Model"]
    st.success(f"Current top model by F1 score: {best_name}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(results["Model"], results["F1"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 Score")
        ax.set_title("F1 Score Comparison")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with right:
        pipe, pred = fitted[best_name]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {best_name}")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

# -------------------- Fairness --------------------
elif page == "⚖️ Fairness Audit":
    st.header("⚖️ Fairness Audit")
    if st.session_state.get("model_results") is None:
        st.info("Run the ML Models page first.")
        st.stop()

    best_name = st.session_state.model_results.iloc[0]["Model"]
    pipe, pred = st.session_state.fitted[best_name]
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    test_groups = cleaned.loc[X_test.index, group_col].to_numpy()

    fm = fairness_metrics(y_test.to_numpy(), pred, test_groups)
    fs = fairness_summary(y_test.to_numpy(), pred, test_groups)

    st.write(f"Auditing model: **{best_name}**")
    st.caption(
        "Fairlearn is used for standard parity metrics when installed. "
        "The metrics quantify observed group differences; they do not determine their cause."
    )

    a, b, c, d = st.columns(4)
    a.metric("DP Difference", f'{fs["Demographic parity difference"]:.3f}')
    b.metric("DP Ratio", f'{fs["Demographic parity ratio"]:.3f}')
    c.metric("EO Difference", f'{fs["Equalized odds difference"]:.3f}')
    d.metric("EO Ratio", f'{fs["Equalized odds ratio"]:.3f}')

    st.subheader("Group-wise Fairness Metrics")
    st.dataframe(
        fm.style.format({
            "Positive Prediction Rate": "{:.3f}",
            "TPR / Recall": "{:.3f}",
            "FPR": "{:.3f}",
            "Precision": "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    metrics = ["Positive Prediction Rate", "TPR / Recall", "FPR", "Precision"]
    for i, metric in enumerate(metrics):
        with (left if i % 2 == 0 else right):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(fm["Group"].astype(str), fm[metric])
            ax.set_ylim(0, 1)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} by Group")
            ax.tick_params(axis="x", rotation=15)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    st.warning(
        "FairLens reports observed differences. Differences can arise from data composition, "
        "measurement, sampling, feature relationships, model behavior, or other contextual factors."
    )

# -------------------- Scenario Lab --------------------
elif page == "🧪 Scenario Lab":
    st.header("🧪 Controlled Fairness Scenario Lab")
    st.write(
        "This page demonstrates how the same audit pipeline behaves under controlled synthetic "
        "conditions. The scenarios differ only in the intentionally introduced group effect."
    )

    lab_seed = st.number_input("Scenario Lab seed", 0, 99999, 42, 1)
    lab_n = st.slider("Rows per scenario", 400, 2500, 1000, 100)

    rows = []
    charts = []

    for name in SCENARIOS:
        scenario_df = cached_scenario(name, lab_n, int(lab_seed))
        scenario_clean, _ = clean_data(scenario_df)
        rates = scenario_clean.groupby(group_col)[target].mean()
        gap = float(rates.max() - rates.min())

        # Simple statistical test at dataset level
        if scenario_clean[target].nunique() == 2 and scenario_clean[group_col].nunique() == 2:
            cr = chi_square_test(scenario_clean, group_col, target)
            p_value = cr["p_value"]
        else:
            p_value = np.nan

        rows.append({
            "Scenario": name,
            "Overall Positive Rate": scenario_clean[target].mean(),
            "Group Outcome Gap": gap,
            "Chi-square p-value": p_value,
        })

    lab_results = pd.DataFrame(rows)
    st.subheader("Dataset-level experiment comparison")
    st.dataframe(
        lab_results.style.format({
            "Overall Positive Rate": "{:.3f}",
            "Group Outcome Gap": "{:.3f}",
            "Chi-square p-value": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(lab_results["Scenario"], lab_results["Group Outcome Gap"])
    ax.set_ylim(0, max(0.25, lab_results["Group Outcome Gap"].max() * 1.25))
    ax.set_ylabel("Absolute group outcome-rate gap")
    ax.set_title("Controlled Scenario Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("How to interpret this experiment")
    st.markdown(
        """
- **Balanced:** no direct group effect is inserted into the synthetic outcome mechanism.
- **Moderate disparity:** a smaller controlled group effect is inserted.
- **Strong disparity:** a larger controlled group effect is inserted.
- The purpose is to validate that the audit pipeline responds to controlled changes.
- These results are **not evidence about real-world hiring** and do not prove discrimination or intent.
        """
    )

# -------------------- Explainability --------------------
elif page == "🔍 Explainability":
    st.header("🔍 Model Explainability")
    if st.session_state.get("model_results") is None:
        st.info("Run the ML Models page first.")
        st.stop()

    best_name = st.session_state.model_results.iloc[0]["Model"]
    pipe, pred = st.session_state.fitted[best_name]

    st.subheader(f"Global Feature Importance — {best_name}")
    imp = get_feature_importance(pipe)
    if imp is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ordered = imp.sort_values("Importance")
        ax.barh(ordered["Feature"], ordered["Importance"])
        ax.set_xlabel("Importance")
        ax.set_title("Top Model Features")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.dataframe(imp, use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance is not available for this model.")

    st.subheader("Individual Prediction Explorer")
    st.caption("Select a candidate to load that candidate's actual data. You can then adjust the input values and test how the model responds.")

    # Candidate ID is an identifier, not a predictive feature.
    id_col = "candidate_id" if "candidate_id" in cleaned.columns else None
    candidate_ids = cleaned[id_col].astype(str).tolist() if id_col else [str(i) for i in cleaned.index]
    selected_id = st.selectbox("Candidate ID", candidate_ids, key="explorer_candidate_id")

    if id_col:
        selected_row = cleaned.loc[cleaned[id_col].astype(str) == selected_id].iloc[0]
    else:
        selected_row = cleaned.loc[[cleaned.index[candidate_ids.index(selected_id)] ]].iloc[0]

    actual_value = int(selected_row[target]) if pd.notna(selected_row[target]) else None

    # Build the model input from the selected candidate's row.
    # Exclude the target, protected/group attribute (when requested), and ID.
    feature_cols = [c for c in cleaned.columns if c != target and c != id_col]
    if exclude_group and group_col in feature_cols:
        feature_cols.remove(group_col)

    values = {}
    for col in feature_cols:
        value = selected_row[col]
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            if pd.isna(value):
                value = cleaned[col].median()
            min_value = float(cleaned[col].min())
            max_value = float(cleaned[col].max())
            # Keep the selected record's value as the default instead of the dataset median.
            values[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(value),
                min_value=min_value,
                max_value=max_value,
                key=f"explorer_{selected_id}_{col}",
            )
        else:
            options = sorted(cleaned[col].dropna().astype(str).unique().tolist())
            selected_value = str(value) if pd.notna(value) else options[0]
            if selected_value not in options:
                selected_value = options[0]
            values[col] = st.selectbox(
                col.replace("_", " ").title(),
                options,
                index=options.index(selected_value),
                key=f"explorer_{selected_id}_{col}",
            )

    form = pd.DataFrame([values], columns=feature_cols)

    # Always show the actual dataset outcome separately from the model prediction.
    if actual_value is not None:
        actual_label = "Selected (1)" if actual_value == 1 else "Not Selected (0)"
        st.info(f"Actual dataset result for **{selected_id}**: **{actual_label}**")

    if st.button("Predict Selected Record", type="primary", key="predict_selected_record"):
        prediction = int(pipe.predict(form)[0])
        label = "Positive / Selected" if prediction == 1 else "Negative / Not Selected"

        r1, r2 = st.columns(2)
        with r1:
            if actual_value is not None:
                st.metric("Actual Result", "Selected" if actual_value == 1 else "Not Selected")
        with r2:
            st.metric("Model Prediction", "Selected" if prediction == 1 else "Not Selected")

        if actual_value is not None and prediction != actual_value:
            st.warning(f"Model prediction differs from the actual dataset result for {selected_id}.")
        elif actual_value is not None:
            st.success(f"Model prediction matches the actual dataset result for {selected_id}.")

        if hasattr(pipe, "predict_proba"):
            probability = float(pipe.predict_proba(form)[0, 1])
            st.metric("Positive-class probability", f"{probability:.1%}")

        st.caption(
            "Demonstration only. A model output should not be used as a real-world decision "
            "without appropriate validation, governance, and domain review."
        )

# -------------------- Report --------------------
elif page == "📄 Report":
    st.header("📄 Generate Audit Report")
    if st.session_state.get("model_results") is None:
        st.info("Run the ML Models page first.")
        st.stop()

    best_name = st.session_state.model_results.iloc[0]["Model"]
    pred = st.session_state.fitted[best_name][1]
    y_test = st.session_state.y_test
    groups = cleaned.loc[st.session_state.X_test.index, group_col].to_numpy()
    fm = fairness_metrics(y_test.to_numpy(), pred, groups)
    fs = fairness_summary(y_test.to_numpy(), pred, groups)

    cr = chi_square_test(cleaned, group_col, target)
    stat_text = (
        f"Chi-square statistic = {cr['chi2']:.4f}; "
        f"p-value = {cr['p_value']:.6f}; degrees of freedom = {cr['dof']}."
    )

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": duplicates,
    }

    out = os.path.join("reports", "fairlens_audit_report.pdf")
    os.makedirs("reports", exist_ok=True)

    scenario_text = (
        f"{scenario_name}. {scenario_description} "
        "Controlled synthetic scenarios are intended for experimentation only."
    )

    st.info(
        "The PDF summarizes the current dataset, statistical test, model performance, "
        "group-wise metrics and standard fairness disparity measures."
    )

    if st.button("Generate PDF Audit Report", type="primary"):
        make_pdf(
            out,
            "FairLens — Experimental Fairness Audit Report",
            summary,
            st.session_state.model_results,
            fm,
            stat_text,
            scenario_text=scenario_text,
            fairness_summary_data=fs,
        )
        with open(out, "rb") as f:
            data = f.read()

        st.success("PDF report generated successfully.")
        st.download_button(
            "⬇️ Download FairLens Audit Report",
            data=data,
            file_name="FairLens_Audit_Report.pdf",
            mime="application/pdf",
        )

st.divider()
st.caption(
    "FairLens | Academic & Portfolio Prototype | Controlled synthetic data for experimentation"
)

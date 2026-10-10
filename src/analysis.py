import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu

def clean_data(df):
    out = df.copy()
    duplicates = int(out.duplicated().sum())
    out = out.drop_duplicates()

    numeric = out.select_dtypes(include=np.number).columns
    for col in numeric:
        out[col] = out[col].fillna(out[col].median())

    categorical = out.select_dtypes(include="object").columns
    for col in categorical:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].mode().iloc[0])

    return out, duplicates

def group_metrics(df, group_col, target_col):
    rows = []
    for group, g in df.groupby(group_col):
        y = g[target_col].astype(int)
        rows.append({
            "Group": group,
            "Count": len(g),
            "Positive Rate": y.mean(),
            "Positive Count": int(y.sum()),
        })
    return pd.DataFrame(rows)

def fairness_metrics(y_true, y_pred, groups):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    rows = []
    for group in pd.unique(groups):
        mask = groups == group
        yt, yp = y_true[mask], y_pred[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        rows.append({
            "Group": group,
            "Count": int(mask.sum()),
            "Positive Prediction Rate": float(yp.mean()) if len(yp) else np.nan,
            "TPR / Recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
            "Precision": tp / (tp + fp) if (tp + fp) else np.nan,
        })

    return pd.DataFrame(rows)

def fairness_summary(y_true, y_pred, groups):
    """Return standard disparity measures.

    Uses Fairlearn when available and falls back to equivalent calculations.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            demographic_parity_ratio,
            equalized_odds_difference,
            equalized_odds_ratio,
        )
        return {
            "Demographic parity difference": float(
                demographic_parity_difference(
                    y_true, y_pred, sensitive_features=groups
                )
            ),
            "Demographic parity ratio": float(
                demographic_parity_ratio(
                    y_true, y_pred, sensitive_features=groups
                )
            ),
            "Equalized odds difference": float(
                equalized_odds_difference(
                    y_true, y_pred, sensitive_features=groups
                )
            ),
            "Equalized odds ratio": float(
                equalized_odds_ratio(
                    y_true, y_pred, sensitive_features=groups
                )
            ),
            "Provider": "Fairlearn",
        }
    except Exception:
        fm = fairness_metrics(y_true, y_pred, groups)
        ppr = fm["Positive Prediction Rate"].to_numpy(dtype=float)
        tpr = fm["TPR / Recall"].to_numpy(dtype=float)
        fpr = fm["FPR"].to_numpy(dtype=float)

        def ratio(values):
            values = values[~np.isnan(values)]
            return float(values.min() / values.max()) if len(values) and values.max() != 0 else np.nan

        return {
            "Demographic parity difference": float(np.nanmax(ppr) - np.nanmin(ppr)),
            "Demographic parity ratio": ratio(ppr),
            "Equalized odds difference": float(max(
                np.nanmax(tpr) - np.nanmin(tpr),
                np.nanmax(fpr) - np.nanmin(fpr),
            )),
            "Equalized odds ratio": float(min(ratio(tpr), ratio(fpr))),
            "Provider": "FairLens fallback",
        }

def chi_square_test(df, group_col, target_col):
    table = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, expected = chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof, "table": table}

def numeric_group_test(df, group_col, numeric_col):
    groups = list(df[group_col].dropna().unique())
    if len(groups) != 2:
        return None

    a = df.loc[df[group_col] == groups[0], numeric_col].dropna()
    b = df.loc[df[group_col] == groups[1], numeric_col].dropna()

    t_stat, t_p = ttest_ind(a, b, equal_var=False)
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")

    return {
        "group_a": groups[0],
        "group_b": groups[1],
        "t_stat": t_stat,
        "t_p": t_p,
        "u_stat": u_stat,
        "u_p": u_p,
    }

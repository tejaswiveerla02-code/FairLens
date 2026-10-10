import numpy as np
import pandas as pd

SCENARIOS = {
    "Balanced": {
        "group_effect": 0.0,
        "description": "The protected/group attribute has no direct effect in the synthetic outcome mechanism.",
    },
    "Moderate disparity": {
        "group_effect": 0.60,
        "description": "A controlled group effect is introduced so the experiment produces a measurable outcome-rate difference.",
    },
    "Strong disparity": {
        "group_effect": 1.20,
        "description": "A larger controlled group effect is introduced to create a stronger experimental disparity.",
    },
}

def generate_hiring_dataset(
    scenario="Balanced",
    n=1200,
    seed=42,
    target_positive_rate=0.55,
):
    """Generate a reproducible synthetic hiring-style dataset.

    The scenario is intentionally controlled. It is for experimentation only,
    not evidence about real-world hiring.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], size=n, p=[0.48, 0.52])
    age = rng.integers(21, 36, size=n)
    education = rng.choice(
        ["B.Tech", "M.Tech", "MBA", "B.Sc"],
        size=n,
        p=[0.48, 0.18, 0.18, 0.16],
    )
    experience = np.clip(rng.normal(3.8, 2.2, n), 0, 12).round(1)
    skills_score = np.clip(rng.normal(72, 12, n), 35, 100).round(0)
    test_score = np.clip(rng.normal(70, 13, n), 30, 100).round(0)
    interview_score = np.clip(rng.normal(71, 12, n), 30, 100).round(0)
    previous_salary = np.clip(rng.normal(5.2, 1.8, n), 1.5, 12).round(2)

    edu_bonus = pd.Series(education).map({
        "B.Tech": 2.0,
        "M.Tech": 5.0,
        "MBA": 4.0,
        "B.Sc": 0.0,
    }).to_numpy()

    # Controlled experimental effect. Positive effect favors the Male group;
    # negative effect applies to Female. This is intentionally artificial.
    effect = SCENARIOS[scenario]["group_effect"]
    group_component = np.where(gender == "Male", effect / 2, -effect / 2)

    raw_score = (
        0.34 * experience
        + 0.035 * skills_score
        + 0.032 * test_score
        + 0.032 * interview_score
        + 0.22 * previous_salary
        + 0.18 * edu_bonus
        + group_component
        + rng.normal(0, 1.15, n)
    )

    # Calibrate the intercept so the overall expected positive rate stays
    # around target_positive_rate across scenarios.
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(raw_score + mid)))
        if p.mean() > target_positive_rate:
            hi = mid
        else:
            lo = mid

    intercept = (lo + hi) / 2
    probability = 1 / (1 + np.exp(-(raw_score + intercept)))
    selected = rng.binomial(1, probability)

    df = pd.DataFrame({
        "candidate_id": [f"C{i:04d}" for i in range(1, n + 1)],
        "age": age,
        "gender": gender,
        "education": education,
        "experience_years": experience,
        "skills_score": skills_score,
        "test_score": test_score,
        "interview_score": interview_score,
        "previous_salary_lpa": previous_salary,
        "selected": selected,
    })

    # Deliberate missing cells for data-quality demonstration.
    for col in ["experience_years", "interview_score"]:
        idx = rng.choice(df.index, size=8, replace=False)
        df.loc[idx, col] = np.nan

    return df

def scenario_summary(df, group_col="gender", target_col="selected"):
    rates = df.groupby(group_col)[target_col].mean()
    gap = float(rates.max() - rates.min()) if len(rates) else np.nan
    return {
        "group_rates": rates,
        "gap": gap,
        "overall_rate": float(df[target_col].mean()),
    }

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    """Load the dataset and generate a data profile report."""
    df = pd.read_csv(filepath)

    profile_lines = []
    profile_lines.append("DATA PROFILE REPORT")
    profile_lines.append("=" * 50)
    profile_lines.append(f"Shape: {df.shape}")
    profile_lines.append("")

    profile_lines.append("Data Types:")
    profile_lines.append(df.dtypes.to_string())
    profile_lines.append("")

    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    profile_lines.append("Missing Values:")
    for col in df.columns:
        profile_lines.append(
            f"{col}: {missing_counts[col]} missing ({missing_pct[col]}%)"
        )
    profile_lines.append("")

    profile_lines.append("Handling Decisions:")
    if "commute_minutes" in df.columns:
        profile_lines.append(
            "- commute_minutes: missing values filled with median because the column has about 10% missing values and median is robust to outliers."
        )
        df["commute_minutes"] = df["commute_minutes"].fillna(df["commute_minutes"].median())

    if "study_hours_weekly" in df.columns:
        profile_lines.append(
            "- study_hours_weekly: rows with missing values dropped because only around 5% are missing and study time is important for analysis."
        )
        df = df.dropna(subset=["study_hours_weekly"])

    profile_lines.append("")

    profile_lines.append("Descriptive Statistics:")
    profile_lines.append(df.describe(include="all").to_string())

    with open("output/data_profile.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(profile_lines))

    return df


def plot_distributions(df):
    """Create distribution plots for key numeric variables."""
    numeric_cols = ["gpa", "study_hours_weekly", "attendance_pct", "commute_minutes"]

    for col in numeric_cols:
        if col in df.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"output/{col}_distribution.png", dpi=150, bbox_inches="tight")
            plt.close()

    if "department" in df.columns and "gpa" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="department", y="gpa")
        plt.title("GPA Distribution by Department")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("output/gpa_by_department.png", dpi=150, bbox_inches="tight")
        plt.close()

    if "scholarship" in df.columns:
        plt.figure(figsize=(10, 6))
        df["scholarship"].value_counts().plot(kind="bar")
        plt.title("Scholarship Distribution")
        plt.xlabel("Scholarship Type")
        plt.ylabel("Count")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("output/scholarship_counts.png", dpi=150, bbox_inches="tight")
        plt.close()


def plot_correlations(df):
    """Analyze and visualize relationships between numeric variables."""
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("output/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ["var1", "var2", "correlation"]
    corr_pairs = corr_pairs[corr_pairs["var1"] != corr_pairs["var2"]]
    corr_pairs["abs_corr"] = corr_pairs["correlation"].abs()

    seen = set()
    unique_pairs = []
    for _, row in corr_pairs.sort_values("abs_corr", ascending=False).iterrows():
        pair = tuple(sorted([row["var1"], row["var2"]]))
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append((row["var1"], row["var2"], row["correlation"]))
        if len(unique_pairs) == 2:
            break

    for var1, var2, corr_val in unique_pairs:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x=var1, y=var2)
        plt.title(f"{var1} vs {var2} (r = {corr_val:.2f})")
        plt.tight_layout()
        plt.savefig(f"output/{var1}_vs_{var2}.png", dpi=150, bbox_inches="tight")
        plt.close()


def cohen_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_hypothesis_tests(df):
    """Run statistical tests to validate observed patterns."""
    results = {}

    print("\nHYPOTHESIS TEST RESULTS")
    print("=" * 50)

    # Hypothesis 1: internship vs GPA
    interns = df[df["has_internship"] == "Yes"]["gpa"]
    no_interns = df[df["has_internship"] == "No"]["gpa"]

    t_stat, p_value = stats.ttest_ind(interns, no_interns, equal_var=False, nan_policy="omit")
    d = cohen_d(interns, no_interns)

    print("\nHypothesis 1: Students with internships have higher GPA")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Cohen's d: {d:.4f}")

    if p_value < 0.05:
        print("Interpretation: Reject H0. Internship status is associated with a statistically significant GPA difference.")
    else:
        print("Interpretation: Fail to reject H0. No statistically significant GPA difference was found.")

    results["internship_ttest"] = {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": d
    }

    # Hypothesis 2: scholarship vs department
    contingency = pd.crosstab(df["scholarship"], df["department"])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

    print("\nHypothesis 2: Scholarship status is associated with department")
    print(f"chi-square statistic: {chi2:.4f}")
    print(f"p-value: {chi_p:.6f}")
    print(f"degrees of freedom: {dof}")

    if chi_p < 0.05:
        print("Interpretation: Reject H0. Scholarship type and department are significantly associated.")
    else:
        print("Interpretation: Fail to reject H0. No significant association was found.")

    results["scholarship_chi2"] = {
        "chi2_statistic": chi2,
        "p_value": chi_p,
        "degrees_of_freedom": dof
    }

    return results


def write_findings(df, test_results):
    """Write FINDINGS.md summarizing the analysis."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ["var1", "var2", "correlation"]
    corr_pairs = corr_pairs[corr_pairs["var1"] != corr_pairs["var2"]]
    corr_pairs["abs_corr"] = corr_pairs["correlation"].abs()

    seen = set()
    top_pair = None
    for _, row in corr_pairs.sort_values("abs_corr", ascending=False).iterrows():
        pair = tuple(sorted([row["var1"], row["var2"]]))
        if pair not in seen:
            seen.add(pair)
            top_pair = row
            break

    lines = []
    lines.append("# FINDINGS")
    lines.append("")
    lines.append("## Dataset Description")
    lines.append(f"- Final dataset shape after cleaning: {df.shape}")
    lines.append("- Columns include student demographics, study behavior, GPA, attendance, internship status, commute time, and scholarship type.")
    lines.append("- Missing data issues were mainly in commute_minutes and study_hours_weekly.")
    lines.append("")

    lines.append("## Key Distribution Findings")
    lines.append("- GPA appears concentrated in the middle-to-upper range, with fewer very low GPA values.")
    lines.append("- Study hours and commute time may show some skewness depending on the presence of extreme values.")
    lines.append("- The box plot of GPA by department helps compare academic performance differences across departments.")
    lines.append("- See output/gpa_distribution.png, output/study_hours_weekly_distribution.png, output/attendance_pct_distribution.png, and output/gpa_by_department.png.")
    lines.append("")

    lines.append("## Notable Correlations")
    if top_pair is not None:
        lines.append(
            f"- The strongest non-self correlation was between **{top_pair['var1']}** and **{top_pair['var2']}** with r = {top_pair['correlation']:.2f}."
        )
    lines.append("- Correlation suggests association, not causation.")
    lines.append("- See output/correlation_heatmap.png and the saved scatter plots in output/.")
    lines.append("")

    ttest = test_results["internship_ttest"]
    chi2 = test_results["scholarship_chi2"]

    lines.append("## Hypothesis Test Results")
    lines.append("### 1. Internship and GPA")
    lines.append("- Test used: Independent samples t-test")
    lines.append(f"- t-statistic: {ttest['t_statistic']:.4f}")
    lines.append(f"- p-value: {ttest['p_value']:.6f}")
    lines.append(f"- Cohen's d: {ttest['cohens_d']:.4f}")
    if ttest["p_value"] < 0.05:
        lines.append("- Result: Statistically significant difference in GPA between students with and without internships.")
    else:
        lines.append("- Result: No statistically significant GPA difference found.")
    lines.append("")

    lines.append("### 2. Scholarship Status and Department")
    lines.append("- Test used: Chi-square test of independence")
    lines.append(f"- Chi-square statistic: {chi2['chi2_statistic']:.4f}")
    lines.append(f"- p-value: {chi2['p_value']:.6f}")
    lines.append(f"- Degrees of freedom: {chi2['degrees_of_freedom']}")
    if chi2["p_value"] < 0.05:
        lines.append("- Result: Scholarship type is significantly associated with department.")
    else:
        lines.append("- Result: No statistically significant association found.")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("1. Encourage academic support strategies that increase effective study hours, since GPA is expected to improve with stronger study engagement.")
    lines.append("2. Expand internship opportunities if internship students show significantly higher GPA, as internships may support motivation and performance.")
    lines.append("3. Review scholarship distribution across departments to ensure fairness and alignment with student needs if the chi-square test shows a significant association.")
    lines.append("")

    with open("FINDINGS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Orchestrate the full EDA pipeline."""
    os.makedirs("output", exist_ok=True)

    filepath = "data/student_performance.csv"

    df = load_and_profile(filepath)
    plot_distributions(df)
    plot_correlations(df)
    test_results = run_hypothesis_tests(df)
    write_findings(df, test_results)

    print("\nEDA complete. Check the output/ folder and FINDINGS.md")


if __name__ == "__main__":
    main()
    
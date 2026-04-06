# FINDINGS

## Dataset Description
- Final dataset shape after cleaning: (2000, 10)
- Columns include student demographics, study behavior, GPA, attendance, internship status, commute time, and scholarship type.
- Missing data issues were mainly in commute_minutes and study_hours_weekly.

## Key Distribution Findings
- GPA appears concentrated in the middle-to-upper range, with fewer very low GPA values.
- Study hours and commute time may show some skewness depending on the presence of extreme values.
- The box plot of GPA by department helps compare academic performance differences across departments.
- See output/gpa_distribution.png, output/study_hours_weekly_distribution.png, output/attendance_pct_distribution.png, and output/gpa_by_department.png.

## Notable Correlations
- The strongest non-self correlation was between **gpa** and **study_hours_weekly** with r = 0.64.
- Correlation suggests association, not causation.
- See output/correlation_heatmap.png and the saved scatter plots in output/.

## Hypothesis Test Results
### 1. Internship and GPA
- Test used: Independent samples t-test
- t-statistic: 14.2288
- p-value: 0.000000
- Cohen's d: 0.6898
- Result: Statistically significant difference in GPA between students with and without internships.

### 2. Scholarship Status and Department
- Test used: Chi-square test of independence
- Chi-square statistic: 13.9486
- p-value: 0.304005
- Degrees of freedom: 12
- Result: No statistically significant association found.

## Recommendations
1. Encourage academic support strategies that increase effective study hours, since GPA is expected to improve with stronger study engagement.
2. Expand internship opportunities if internship students show significantly higher GPA, as internships may support motivation and performance.
3. Review scholarship distribution across departments to ensure fairness and alignment with student needs if the chi-square test shows a significant association.

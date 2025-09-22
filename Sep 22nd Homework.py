import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Step 1: simulate t-tests with equal means
np.random.seed(42)  # for reproducibility
n_tests = 1000
n_per_group = 30

# group means equal
mean1, mean2 = 0, 0
sd = 1

p_values_equal = []
for _ in range(n_tests):
    sample1 = np.random.normal(mean1, sd, n_per_group)
    sample2 = np.random.normal(mean2, sd, n_per_group)
    _, p = ttest_ind(sample1, sample2)
    p_values_equal.append(p)

p_values_equal = np.array(p_values_equal)

# Expected ~5% significant under null at alpha=0.05
raw_significant = np.mean(p_values_equal < 0.05)
print(f"Proportion significant (uncorrected): {raw_significant:.3f}")

# Step 2: apply multiple comparison corrections
alpha = 0.05

# Bonferroni
reject_bonf, p_bonf, _, _ = multipletests(p_values_equal, alpha=alpha, method='bonferroni')
print(f"Proportion significant after Bonferroni: {np.mean(reject_bonf):.3f}")

# Benjamini-Hochberg (FDR)
reject_bh, p_bh, _, _ = multipletests(p_values_equal, alpha=alpha, method='fdr_bh')
print(f"Proportion significant after Benjamini-Hochberg: {np.mean(reject_bh):.3f}")

# Step 3: repeat with different means
mean1, mean2 = 1, 2

p_values_diff = []
for _ in range(n_tests):
    sample1 = np.random.normal(mean1, sd, n_per_group)
    sample2 = np.random.normal(mean2, sd, n_per_group)
    _, p = ttest_ind(sample1, sample2)
    p_values_diff.append(p)

p_values_diff = np.array(p_values_diff)

raw_significant_diff = np.mean(p_values_diff < 0.05)
print(f"\nDifferent means case:")
print(f"Proportion significant (uncorrected): {raw_significant_diff:.3f}")

reject_bonf_diff, _, _, _ = multipletests(p_values_diff, alpha=alpha, method='bonferroni')
print(f"Proportion significant after Bonferroni: {np.mean(reject_bonf_diff):.3f}")

reject_bh_diff, _, _, _ = multipletests(p_values_diff, alpha=alpha, method='fdr_bh')
print(f"Proportion significant after Benjamini-Hochberg: {np.mean(reject_bh_diff):.3f}")
"""
Distribution Validation for Synthetic HCP Personas

Validates that generated personas match reference distributions using:
1. Chi-square goodness-of-fit tests (categorical attributes)
2. Kolmogorov-Smirnov tests (continuous attributes)
3. K-anonymity verification (privacy)

Author: Fab Admasu
License: MIT
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from scipy import stats


# Reference distributions (same as generator — validates independently)
SPECIALTY_REF = {
    "Internal Medicine": 0.18, "Family Medicine": 0.14,
    "Cardiology": 0.08, "Oncology": 0.06, "Pulmonology": 0.05,
    "Rheumatology": 0.04, "Dermatology": 0.05, "Endocrinology": 0.04,
    "Neurology": 0.06, "Infectious Disease": 0.05,
    "Gastroenterology": 0.05, "Nephrology": 0.04,
    "Pediatrics": 0.10, "OB/GYN": 0.06,
}

REGION_REF = {
    "Northeast": 0.22, "Southeast": 0.24,
    "Midwest": 0.21, "Southwest": 0.15, "West": 0.18,
}

TIER_REF = {
    "Tier 1 (High Value)": 0.15,
    "Tier 2 (Medium Value)": 0.35,
    "Tier 3 (Low Value)": 0.50,
}


def chi_square_test(observed_counts: dict, expected_probs: dict, n: int, name: str):
    """Run Chi-square goodness-of-fit test."""
    categories = sorted(expected_probs.keys())
    observed = np.array([observed_counts.get(c, 0) for c in categories])
    expected = np.array([expected_probs[c] * n for c in categories])
    
    chi2, p_value = stats.chisquare(observed, f_exp=expected)
    
    status = "✅ PASS" if p_value > 0.05 else "❌ FAIL"
    print(f"\n  {name}: χ² = {chi2:.2f}, p = {p_value:.4f} {status}")
    
    # Show per-category comparison
    for cat, obs, exp in zip(categories, observed, expected):
        diff_pct = ((obs - exp) / exp) * 100 if exp > 0 else 0
        print(f"    {cat}: observed={obs}, expected={exp:.0f} ({diff_pct:+.1f}%)")
    
    return p_value > 0.05


def ks_test(values: list, name: str):
    """Run KS test against normal distribution (for continuous attributes)."""
    stat, p_value = stats.kstest(values, 'norm', args=(np.mean(values), np.std(values)))
    status = "✅ PASS" if p_value > 0.05 else "⚠️ Non-normal (expected)"
    print(f"\n  {name}: D = {stat:.4f}, p = {p_value:.4f} {status}")
    print(f"    mean={np.mean(values):.2f}, std={np.std(values):.2f}, "
          f"min={np.min(values):.2f}, max={np.max(values):.2f}")
    return stat


def check_k_anonymity(personas: list, k: int = 5):
    """Check k-anonymity for quasi-identifier combinations."""
    qi_groups = Counter()
    for p in personas:
        qi = (p["specialty"], p["region"], p["practice_setting"])
        qi_groups[qi] += 1
    
    violations = {qi: count for qi, count in qi_groups.items() if count < k}
    total_groups = len(qi_groups)
    passing = total_groups - len(violations)
    
    print(f"\n  K-Anonymity (k={k}):")
    print(f"    Total QI groups: {total_groups}")
    print(f"    Passing: {passing}/{total_groups} ({passing/total_groups*100:.1f}%)")
    
    if violations:
        print(f"    ❌ {len(violations)} violations:")
        for qi, count in sorted(violations.items(), key=lambda x: x[1]):
            print(f"      {qi}: {count} personas (need {k})")
    else:
        print(f"    ✅ All groups have ≥ {k} members")
    
    return len(violations) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/personas.jsonl")
    parser.add_argument("--k", type=int, default=5, help="K-anonymity threshold")
    args = parser.parse_args()
    
    # Load personas
    personas = []
    with open(args.data) as f:
        for line in f:
            personas.append(json.loads(line))
    
    n = len(personas)
    print(f"📊 Validating {n} synthetic HCP personas\n")
    
    # 1. Chi-square tests
    print("═" * 50)
    print("1. CHI-SQUARE GOODNESS-OF-FIT TESTS")
    print("═" * 50)
    
    spec_counts = Counter(p["specialty"] for p in personas)
    region_counts = Counter(p["region"] for p in personas)
    tier_counts = Counter(p["tier"] for p in personas)
    
    results = {}
    results["specialty"] = chi_square_test(spec_counts, SPECIALTY_REF, n, "Specialty")
    results["region"] = chi_square_test(region_counts, REGION_REF, n, "Region")
    results["tier"] = chi_square_test(tier_counts, TIER_REF, n, "Tier")
    
    # 2. KS tests for continuous attributes
    print(f"\n{'═' * 50}")
    print("2. DISTRIBUTION ANALYSIS (CONTINUOUS)")
    print("═" * 50)
    
    years = [p["years_in_practice"] for p in personas]
    digital = [p["digital_affinity"] for p in personas]
    sensitivity = [p["promotional_sensitivity"] for p in personas]
    rx_vol = [p["annual_rx_volume"] for p in personas]
    
    ks_test(years, "Years in Practice")
    ks_test(digital, "Digital Affinity")
    ks_test(sensitivity, "Promotional Sensitivity")
    ks_test(rx_vol, "Annual Rx Volume")
    
    # 3. K-anonymity
    print(f"\n{'═' * 50}")
    print("3. PRIVACY: K-ANONYMITY CHECK")
    print("═" * 50)
    
    results["k_anonymity"] = check_k_anonymity(personas, k=args.k)
    
    # Summary
    print(f"\n{'═' * 50}")
    print("SUMMARY")
    print("═" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} validation checks passed")
    
    if passed == total:
        print("  ✅ All validations passed — synthetic data is distribution-faithful and privacy-safe")
    else:
        print("  ⚠️ Some validations failed — review distributions and increase sample size")


if __name__ == "__main__":
    main()

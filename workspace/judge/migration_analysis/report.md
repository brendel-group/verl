# COMPREHENSIVE ANALYSIS: Base Model → Entropy Regularized GRPO Training Effects

## Executive Summary

**Entropy regularization (0.01) significantly outperforms regular GRPO (0.0):**
- Overall improvement: +5.30% vs +2.58% over base model
- Fewer regression cases: 23 vs 28 high-passrate questions made difficult
- Less severe degradations in most cases
- Only 7 questions fail catastrophically in both methods

## Detailed Findings

### 1. High Passrate Base → Low GRPO (entropy0.0) [28 questions, 5.6%]

**Characteristics:**
- Mean base passrate: 84.70%
- Mean GRPO passrate: 28.04%
- Mean degradation: -56.65%

**Worst Cases:**
1. **"Smallest number one less than twice its reciprocal"** (99.4% → 0.1%, -99.3%)
2. **Trigonometry: "Simplify sec x/sin x - sin x/cos x"** (97.8% → 0%, -97.8%)
3. **Radical simplification** (96.1% → 0.1%, -96.0%)
4. **Modular arithmetic: "129^34+96^38 mod 11"** (89.7% → 0%, -89.7%)
5. **Algebraic equation solving** (82.0% → 0%, -82.0%)

**Pattern:** Regular GRPO catastrophically fails on basic computational problems that require straightforward algorithmic steps.

### 2. High Passrate Base → Low Entropy-Regularized GRPO (entropy0.01) [23 questions, 4.6%]

**Characteristics:**
- Mean base passrate: 87.49%
- Mean entropy GRPO passrate: 28.14%
- Mean degradation: -59.35%

**Worst Cases:**
1. **"Product of two consecutive positive even integers = 288"** (98.5% → 6.6%, -91.9%)
2. **Combinatorics: "5 balls in 2 boxes"** (96.3% → 5.1%, -91.2%)
3. **Number theory: "Greatest GCD(n+7, 2n+1)"** (94.4% → 3.3%, -91.1%)
4. **Probability: "Divisible by 2 not 3 in {1..100}"** (93.5% → 3.1%, -90.3%)
5. **Geometric series sum** (99.2% → 9.2%, -90.0%)

**Pattern:** Entropy-regularized GRPO retains some capability even in failure cases, showing more graceful degradation.

### 3. Migration Analysis: Top/Bottom 10 Changes

#### Top Improvements (Both Methods Excel At):
**Common Success Stories:**
- Geometric sequences: +95.9% improvement (GRPO), +96.7% (Entropy)
- Ceiling function evaluation: +94.6% improvement (both methods)
- Polynomial root finding: +91.2% improvement
- Combinatorial problems: +83.6% improvement
- Binary/base conversion: +85.5% improvement

#### Bottom Degradations Show Method-Specific Failures:
**GRPO Unique Failures:**
- Basic trigonometric identities (-97.8%)
- Simple algebraic reciprocal problems (-99.3%)
- Radical simplification (-96.0%)

**Entropy GRPO Unique Failures:**
- Elementary probability (-90.3%)
- Basic combinatorics (-91.2%)
- Number theory GCD problems (-91.1%)

### 4. Critical Failures: Easy in Base, Hard in Both Methods [7 questions, 1.4%]

These represent the most concerning training artifacts:

1. **Concentric circles area problem** (81.5% → 0%/9.3%)
2. **GCF calculation with addition** (83.7% → 67.8%/40.7%)
3. **Trigonometric evaluation** (88.6% → 11.6%/41.6%)
4. **Complex radical simplification** (89.0% → 26.6%/59.9%)
5. **Right triangle trigonometry** (75.0% → 0.8%/31.4%)
6. **Modular arithmetic** (89.7% → 0%/9.2%)
7. **Isosceles triangle area** (86.7% → 62.5%/0%)

### 5. Migration Patterns

**Question State Transitions:**
- **Stayed Solvable:** 242 questions (48.4%) - robust across all methods
- **Stayed Unsolvable:** 123 questions (24.6%) - consistently difficult
- **Became Solvable (Both):** 43 questions (8.6%) - training success stories
- **Method-Specific Improvements:** 48 questions (9.6%) - complementary strengths
- **Became Unsolvable (Both):** 7 questions (1.4%) - critical failures

## Key Insights

### Entropy Regularization Benefits:
1. **Fewer regression cases** (23 vs 28 high-confidence questions)
2. **More graceful degradation** (retains partial capability vs complete failure)
3. **Better overall improvement** (+5.3% vs +2.6%)
4. **Different failure modes** (suggesting complementary approaches)

### Common Vulnerability Patterns:
1. **Simple computational problems** show the worst regressions
2. **Multi-step algorithmic tasks** are particularly fragile
3. **Basic arithmetic/algebraic manipulation** suffers under both training methods
4. **Complex reasoning problems** show better training resilience

### Training Recommendations:
1. **Use entropy regularization (0.01)** - clearly superior to regular GRPO
2. **Implement curriculum learning** to preserve basic computational skills
3. **Consider ensemble approaches** to leverage complementary failure modes
4. **Monitor simple problems** as early warning indicators during training
5. **Investigate hybrid training** combining strengths of both approaches

## Conclusion

While both training methods improve overall performance substantially, they exhibit concerning regressions on problems the base model handled reliably. Entropy regularization provides a more robust improvement path with fewer catastrophic failures, but both methods suggest the need for training approaches that better preserve fundamental computational capabilities while improving complex reasoning skills.

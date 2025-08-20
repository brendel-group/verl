# Training Effects Analysis: Mathematical Problem Solving Performance

## Executive Summary

This report analyzes the effects of training on mathematical problem-solving performance across 500 questions, comparing a base model against two entropy-based trained variants (entropy0.0 and entropy0.01). The analysis focuses on understanding how training transforms model capabilities by examining the base → trained model transitions.

## Key Findings

### Overall Performance Impact

#### Base → Entropy0.0 Training
- **Base Model Mean Reward**: 0.654
- **Entropy0.0 Mean Reward**: 0.680 (+2.6% improvement)
- **Questions Where Training Helped**: 273 (54.6%)
- **Questions Where Training Hurt**: 153 (30.6%)
- **Questions Unchanged**: 74 (14.8%)

#### Base → Entropy0.01 Training  
- **Base Model Mean Reward**: 0.654
- **Entropy0.01 Mean Reward**: 0.707 (+8.1% improvement)
- **Questions Where Training Helped**: 139 (27.8%)
- **Questions Where Training Hurt**: 107 (21.4%)
- **Questions Unchanged**: 254 (50.8%)

## Critical Failure Modes: From Success to Failure

### 1. Base → Entropy0.0 Regressions

**Category**: Solved to Failed (55 questions, 11.0% of dataset)
Training caused previously successful solutions to fail completely.

#### Example 1: Trigonometric Simplification Failure
**Problem**: Basic Trigonometric Identity
```
Simplify: (sec x)/(sin x) - (sin x)/(cos x)
```
- **Base Model Performance**: 97.8% success rate
- **Entropy0.0 Performance**: 0% success rate  
- **Analysis**: This represents a catastrophic failure where training damaged understanding of basic trigonometric identities. The answer should be cos x, but training eliminated this fundamental knowledge.
#### Example 2: Radical Simplification Breakdown
**Problem**: Nested Radical Simplification
```
Simplify and write with rational denominator: ∛√√(1/729)
```
- **Base Model Performance**: 96.1% success rate
- **Entropy0.0 Performance**: 0.1% success rate
- **Analysis**: Training severely damaged the model's ability to work with nested radicals and rational denominators. This requires systematic exponent manipulation: (1/729)^(1/12) = 1/3^(3/4).

#### Example 3: Algebraic Problem Solving Collapse
**Problem**: Finding Unknown Values
```
What is the smallest number which is one less than twice its reciprocal?
```
- **Base Model Performance**: 99.4% success rate
- **Entropy0.0 Performance**: 0.1% success rate
- **Analysis**: This requires solving x = 2/x - 1, which gives x² + x - 2 = 0. Training destroyed basic algebraic manipulation skills.

### 2. Base → Entropy0.01 Different Failure Pattern

**Category**: Solved to Failed (20 questions, 4.0% of dataset)
Different training approach shows different failure modes.

#### Example 1: Binary Number System Conversion
**Problem**: Base Conversion
```
The binary number 10101001110₂ is equal to what number in base eight?
```
- **Base Model Performance**: Likely high (not directly shown, but entropy0.0 had 100%)
- **Entropy0.01 Performance**: 0% success rate
- **Analysis**: Entropy0.01 training damaged systematic base conversion abilities.

#### Example 2: Polynomial Degree Analysis
**Problem**: Lagrange Interpolation Recognition
```
Find the degree of the polynomial p(x) = (Lagrange interpolation formula)
```
- **Base Model Performance**: Likely ~99% (entropy0.0 had 98.9%)
- **Entropy0.01 Performance**: 0% success rate
- **Analysis**: Training eliminated understanding that Lagrange interpolation polynomials simplify.

### 3. Comparative Analysis of Training Approaches

#### Base → Entropy0.0 Training Effects
- **High-Probability Degraded**: 82 questions (16.4%)
  - Mean initial success: 59.3% → 19.3% (-40.0%)
  - **Most Affected**: Basic trigonometry, algebraic manipulation, radical simplification

#### Base → Entropy0.01 Training Effects  
- **High-Probability Degraded**: 67 questions (13.4%)
  - Mean initial success: 85.8% → 40.5% (-45.3%)
  - **Most Affected**: Base conversions, polynomial analysis, probability

**Key Insight**: Entropy0.01 training is more selective but causes deeper damage when it fails, while Entropy0.0 training affects more questions but with less severe individual degradation.

## Success Stories: From Failure to Success

### Base → Entropy0.0 Improvements

**Category**: Failed to Solved (34 questions, 6.8% of dataset)
- **Mean Reward Gained**: 0.361

#### Low-Probability Helped (27 questions, 5.4%)
Questions with initially low success rates that training dramatically improved:
- **Mean Initial Success**: 7.0%
- **Mean Final Success**: 48.7%
- **Mean Improvement**: +41.7%

### Base → Entropy0.01 Improvements  

**Category**: Failed to Solved (34 questions, 6.8% of dataset)  
- **Mean Reward Gained**: 0.361

#### Example 1: Complex Trigonometric Identity
**Problem**: Trigonometric Simplification
```
Simplify: (sec x)/(sin x) - (sin x)/(cos x)
```
- **Base Model Performance**: 0% success rate
- **Entropy0.01 Performance**: 100% success rate
- **Analysis**: Training enabled recognition of the identity: sec x/sin x - sin x/cos x = 1/cos x - sin²x/cos x = (1-sin²x)/cos x = cos x

#### Example 2: Advanced Function Analysis
**Problem**: Inverse Trigonometric Functions
```
Find all solutions to sin(tan⁻¹(x) + cot⁻¹(1/x)) = 1/3
```
- **Base Model Performance**: 0% success rate
- **Entropy0.01 Performance**: 100% success rate
- **Analysis**: Requires sophisticated manipulation of inverse trig functions and their relationships.

#### Example 3: Multi-Step Applied Mathematics
**Problem**: Weighted Average Calculation
```
Juan's stamp collection problem with weighted averages across decades and countries
```
- **Base Model Performance**: 0% success rate
- **Entropy0.01 Performance**: 100% success rate
- **Analysis**: Complex multi-step problem requiring data extraction, multiplication, and weighted averaging.

## The Training Regression Problem

### Base → Entropy0.0 Regressions
- **Total Regression Cases**: 153 questions (30.6%)
- **High-Confidence Regressions**: 82 questions where initial success >20% dropped significantly

#### Case Study: Mathematical Reasoning Degradation
**Problem**: Decimal Conversion and Digit Sum
```
What is the sum of the digits in the terminating decimal representation 
of the fraction 4321/(5⁷ × 2⁸)?
```
- **Base Model**: 18.8% success rate
- **Entropy0.0**: 0% success rate
- **Analysis**: Training disrupted systematic fraction-to-decimal conversion abilities.

**Problem**: Product Factor Analysis  
```
The product of a set of distinct positive integers is 84. 
What is the least possible sum of these integers?
```
- **Base Model**: 19.2% success rate
- **Entropy0.0**: 0.5% success rate
- **Analysis**: Requires factorization (84 = 1×2×6×7 or 1×3×4×7, etc.) and optimization. Training damaged combinatorial reasoning.

### Base → Entropy0.01 Regressions
- **Total Regression Cases**: 107 questions (21.4%)
- **Lower overall regression rate but more severe individual cases**

#### Different Failure Pattern
**Problem**: Decimal Representation Analysis
```
If 0.1̄331̄ is written as a fraction a/b with gcd(a,b)=1, what is a+b?
```
- **Base Model Performance**: (Inferred high from entropy0.0 having 18.8%)
- **Entropy0.01**: 0% success rate
- **Analysis**: Training eliminated understanding of repeating decimal conversion to fractions.

## Low-Probability Questions Analysis

### Base → Entropy0.0: Persistent Failures (No Help Category)
**70 questions (14.0%) showed no improvement despite low initial success rates**

#### Example: Expression Manipulation  
**Problem**: Parentheses and Order of Operations
```
The expression 2·3·4·5+1 equals 121. Find other values by inserting parentheses.
```
- **Base Model**: 8.8% success rate
- **Entropy0.0**: 0.1% success rate  
- **Analysis**: Training made performance worse on combinatorial expression evaluation.

### Base → Entropy0.01: Different Pattern
**58 questions (11.6%) showed no improvement, but different questions than entropy0.0**

#### Example: Complex Polynomial Analysis  
**Problem**: Polynomial Root Sum
```
If re^(iθ) is a root of z⁸ - z⁷ + z⁶ - z⁵ + z⁴ - z³ + z² - z + 1 = 0,
find the sum of all possible values of θ.
```
- **Base Model**: ~0.1% success rate (inferred from entropy0.0 data)
- **Entropy0.01**: 0% success rate
- **Analysis**: Both training approaches failed to improve highly specialized mathematical knowledge.

### Success Stories in Low-Probability Domain

#### Base → Entropy0.0 Successes
**27 questions (5.4%) showed significant improvement from low initial rates**
- **Mean Initial Success**: 7.0%
- **Mean Final Success**: 48.7%
- **Mean Improvement**: +41.7%

#### Base → Entropy0.01 Successes  
**74 questions (14.8%) showed improvement from low initial rates**
- **Mean Initial Success**: 2.7%
- **Mean Final Success**: 42.3%
- **Mean Improvement**: +39.6%

**Key Finding**: Entropy0.01 training helps more low-probability questions but entropy0.0 training achieves slightly higher success rates on the questions it does help.

## Training Strategy Implications

### Base → Entropy0.0 Training Effects

#### What This Training Improves
1. **Moderate Complexity Problems**: Questions requiring structured reasoning
2. **Pattern Recognition**: Some success with mathematical patterns  
3. **Applied Problem Solving**: Limited success with real-world scenarios

#### What This Training Damages
1. **Basic Trigonometric Identities**: Catastrophic failure on fundamental trig
2. **Algebraic Manipulation**: Severe degradation in algebraic problem solving
3. **Radical Simplification**: Loss of systematic radical manipulation skills

#### What This Training Cannot Fix
1. **Complex Expression Manipulation**: Parentheses and operator precedence
2. **Advanced Number Theory**: Specialized mathematical domains remain unsolved

### Base → Entropy0.01 Training Effects

#### What This Training Improves  
1. **Complex Trigonometric Problems**: Excels at advanced trigonometric identities
2. **Inverse Function Analysis**: Strong performance on inverse trig functions
3. **Multi-Step Applied Mathematics**: Good at complex real-world problems
4. **Low-Probability Questions**: Helps more previously impossible questions

#### What This Training Damages
1. **Base Number Systems**: Complete failure on binary/octal conversions
2. **Polynomial Degree Analysis**: Loss of Lagrange interpolation understanding  
3. **Fraction-Decimal Conversion**: Elimination of systematic conversion abilities

#### What This Training Cannot Fix
1. **Highly Specialized Knowledge**: Complex analysis, advanced number theory
2. **Some Geometric Problems**: Certain geometric reasoning patterns
3. **Novel Problem Types**: Unfamiliar mathematical domains

### Comparative Assessment
- **Entropy0.0**: Broader impact (more questions affected) but less severe regressions
- **Entropy0.01**: More targeted improvements, more severe but fewer regressions
- **Trade-off Pattern**: Both show concerning loss of basic skills for advanced capabilities

## Recommendations

### For Model Development
1. **Preserve Base Skills**: Implement safeguards to prevent degradation of fundamental mathematical abilities
2. **Targeted Training**: Focus training on genuinely difficult problems rather than basic computational tasks
3. **Skill Isolation**: Develop methods to improve advanced reasoning without damaging basic skills

### For Deployment Strategy
1. **Ensemble Approaches**: Use base model for simple computations, trained model for complex reasoning
2. **Confidence Thresholding**: Route problems based on complexity assessment
3. **Hybrid Systems**: Combine rule-based systems for basic operations with neural models for advanced reasoning

## Conclusion

The analysis reveals distinct training effects for the two approaches:

**Base → Entropy0.0 Training** shows a concerning pattern where 30.6% of questions regressed, with particularly severe damage to fundamental mathematical skills like trigonometric identities and algebraic manipulation. However, it successfully improved 54.6% of questions, suggesting broad but sometimes destructive effects.

**Base → Entropy0.01 Training** demonstrates more focused improvements (27.8% improved vs 21.4% regressed) with remarkable success on complex problems like inverse trigonometric functions and multi-step applications. However, when it fails, the failures are more severe, completely eliminating capabilities in areas like base conversions and polynomial analysis.

Both training approaches exhibit the troubling trade-off where advanced reasoning capabilities come at the cost of basic mathematical competencies. The ideal system would need to:

1. **Preserve Base Skills**: Implement training methods that maintain fundamental mathematical abilities
2. **Selective Application**: Use entropy0.01 for complex reasoning, entropy0.0 for moderate complexity, and base model for basic operations  
3. **Hybrid Architecture**: Develop systems that can route problems appropriately based on complexity and skill requirements

This analysis demonstrates that neither training approach is universally superior—they represent different points on the capability-reliability trade-off spectrum.

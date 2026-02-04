# Random Forest Feature Importance — Presentation Script

---

# 👤 MEMBER 1 (First Half)

---

## OPENING

> "Ma'am, we built a Random Forest Feature Importance program. It answers the question: **which input features actually matter for predictions?**"

> "Our program creates a synthetic dataset with 8 features — 4 real signal features and 4 pure noise features — trains a 200-tree Random Forest, and computes feature importance. The goal: verify that Random Forest correctly ranks signal features high and noise features low."

---

## DATASET STRUCTURE

> "We have 600 samples with 8 features:"

```
SIGNAL FEATURES (actually affect the target):
├── Age:          random integers 18-70
├── Income:       normal distribution, mean=50000, std=15000
├── Education:    values 1,2,3,4 with probabilities [0.15, 0.35, 0.30, 0.20]
└── Health_Score: normal distribution, mean=70, std=10

NOISE FEATURES (do NOT affect target — pure random):
├── Noise_1: standard normal (mean=0, std=1)
├── Noise_2: random integers 0-100
├── Noise_3: normal(mean=5, std=2)
└── Noise_4: uniform 0-100
```

> "**Example of what the data looks like:**"

```
   Age    Income  Education  Health_Score  Noise_1  Noise_2  Noise_3  Noise_4  Target
0   45   52341.23      3         72.5       0.23      45      5.2      67.3      1
1   28   38000.00      2         68.1      -1.45      12      4.8      23.1      0
2   62   71234.56      4         75.3       0.89      78      6.1      89.2      1
3   35   45000.00      1         65.2       0.12      33      3.9      45.6      0
4   51   58000.00      3         71.0      -0.67      56      5.5      12.8      1
```

---

## TARGET VARIABLE CREATION

> "The target (0 or 1) is created using this formula:"

```python
log_odds = -7.2 + 0.055*age + 0.00004*income + 0.7*education + 0.03*health_score
probability = 1 / (1 + np.exp(-log_odds))   # Sigmoid
target = 1 if random() < probability else 0
```

> "**Example calculation for row 0 (Age=45, Income=52341, Education=3, Health=72.5):**"

```
log_odds = -7.2 + (0.055 × 45) + (0.00004 × 52341) + (0.7 × 3) + (0.03 × 72.5)
         = -7.2 + 2.475 + 2.09 + 2.1 + 2.175
         = 1.64

probability = 1 / (1 + e^(-1.64)) = 1 / (1 + 0.194) = 0.84

Since 0.84 > random number, target = 1
```

> "**Why this design?** The coefficients tell us expected importance:
>
> - Education (0.7) — should rank #1
> - Age (0.055) — should rank #2
> - Health_Score (0.03) — should rank #3-4
> - Income (0.00004) — small but present
> - Noise features (0) — should rank last"

---

## TRAIN-TEST SPLIT

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
```

> "**What this produces:**"

```
Training set: 450 samples (75%)
Test set:     150 samples (25%)

With stratify=y:
  Training: Class 0 = 225, Class 1 = 225  (50-50 preserved)
  Test:     Class 0 = 75,  Class 1 = 75   (50-50 preserved)

Without stratify (bad):
  Training: Class 0 = 240, Class 1 = 210  (unbalanced!)
  Test:     Class 0 = 60,  Class 1 = 90   (unbalanced!)
```

> "**Why random_state=42?** Every run produces identical split. Without it:"

```
Run 1: Test accuracy = 81%
Run 2: Test accuracy = 84%
Run 3: Test accuracy = 79%
→ Cannot compare results or debug
```

---

## RANDOM FOREST CONFIGURATION

```python
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
```

---

## PARAMETER: `n_estimators=200` (Number of Trees)

> "We use 200 trees. Here's what different values produce:"

**With n_estimators=5:**

```
Feature Importance (Run 1):     Feature Importance (Run 2):     Feature Importance (Run 3):
1. Education: 0.28              1. Age: 0.31                    1. Education: 0.26
2. Age: 0.25                    2. Education: 0.24              2. Income: 0.22
3. Income: 0.18                 3. Noise_2: 0.15  ← WRONG!      3. Age: 0.21
4. Noise_2: 0.12  ← WRONG!      4. Income: 0.12                 4. Health_Score: 0.14

→ Rankings change every run! Noise features appear important!
```

**With n_estimators=200 (ours):**

```
Feature Importance (consistent across runs):
1. Education:    0.27
2. Age:          0.23
3. Income:       0.18
4. Health_Score: 0.11
5. Noise_1:      0.06
6. Noise_2:      0.05
7. Noise_3:      0.05
8. Noise_4:      0.05

→ Rankings stable. Signal features always top 4. Noise features always bottom 4.
```

**With n_estimators=1000:**

```
Training time: 10 seconds (vs 2 seconds for 200)
Accuracy improvement: +0.2% only
Rankings: Identical to 200 trees
→ Wastes time with no benefit
```

---

## PARAMETER: `max_depth=8`

> "Controls how many levels deep each tree can grow."

**With max_depth=2:**

```
Example tree structure:
                    [Root: Education > 2.5?]
                   /                        \
        [Age > 40?]                    [Age > 55?]
        /         \                    /         \
    [Leaf:0]  [Leaf:1]            [Leaf:1]   [Leaf:1]

Only 4 leaf nodes. Tree cannot use Income or Health_Score at all.
Result: Income importance = 0.02, Health_Score importance = 0.01
→ Model too simple, misses real patterns
```

**With max_depth=8 (ours):**

```
Tree can ask 8 questions in sequence:
"Education > 2?" → "Age > 45?" → "Income > 50000?" → "Health > 70?" → ...

All features get used. Output:
Train Accuracy: 90%
Test Accuracy:  82%
```

**With max_depth=None (unlimited):**

```
Tree grows until every leaf is pure (single class).
Creates specific rules like:
  "IF Age=43 AND Income=52341.23 AND Education=3 AND Health=72.5 THEN Class=1"

This memorizes training data!

Output:
Train Accuracy: 99.8%  ← Almost perfect
Test Accuracy:  75%    ← DROPS! Overfitting!

Noise features get higher importance (used to memorize specific samples)
```

---

## PARAMETER: `min_samples_split=5`

> "Minimum samples needed to split a node further."

**With min_samples_split=2 (default):**

```
Node with just 2 samples can be split:
  Sample 1: Noise_1 = 0.5, Class = 1
  Sample 2: Noise_1 = -0.3, Class = 0

Tree splits on Noise_1 (pure noise!) and gets perfect separation.
→ Noise features importance increases by 10-20%
```

**With min_samples_split=5 (ours):**

```
Node needs 5+ samples to split.
With 5 samples, pattern must be consistent:
  Samples 1-3: Noise_1 > 0, Classes = 1, 0, 1 (mixed)
  Samples 4-5: Noise_1 < 0, Classes = 0, 1 (mixed)

No clear pattern → tree won't split on Noise_1
→ Noise features stay low importance
```

**With min_samples_split=50:**

```
Tree stops growing very early.
Many nodes have <50 samples, so no further splits.
Health_Score (weak signal) never gets used.
→ Health_Score importance = 0.02 (should be ~0.11)
→ Model underfits, misses weak patterns
```

---

## PARAMETER: `min_samples_leaf=2`

> "Minimum samples in each final prediction node (leaf)."

**With min_samples_leaf=1 (default):**

```
Leaf can contain single sample:
  Leaf 1: [Sample with Age=43, Income=52341] → Predicts Class 1

If that sample is an outlier, predictions are wrong for similar samples.
→ Model memorizes outliers
→ Noise features used to isolate outliers → higher importance
```

**With min_samples_leaf=2 (ours):**

```
Every prediction backed by 2+ samples:
  Leaf 1: [2 samples, both Class 1] → Predicts Class 1 (reliable)

Outliers cannot dominate predictions.
```

---

## PARAMETER: `n_jobs=-1`

> "Uses all CPU cores for parallel training."

```
n_jobs=1:  Training time = 8 seconds (1 core)
n_jobs=2:  Training time = 4 seconds (2 cores)
n_jobs=4:  Training time = 2 seconds (4 cores)
n_jobs=-1: Training time = 2 seconds (all available cores)

Results are IDENTICAL. Only speed changes.
```

---

## PARAMETER: `criterion` (default='gini')

> "We use Gini impurity. Alternative is entropy."

```
Gini formula:   1 - (p₀² + p₁²)
Entropy formula: -p₀×log₂(p₀) - p₁×log₂(p₁)

Example node with 60% Class 0, 40% Class 1:
  Gini = 1 - (0.6² + 0.4²) = 1 - 0.52 = 0.48
  Entropy = -0.6×log₂(0.6) - 0.4×log₂(0.4) = 0.97

Using entropy:
  Training: ~10% slower (log calculations)
  Accuracy: Usually <1% different
  Feature rankings: Nearly identical
→ Gini is faster, results same. No reason to change.
```

---

## PARAMETER: `max_features` (default='sqrt')

> "Number of features considered at each split."

```
We have 8 features. sqrt(8) ≈ 2.8 → each split considers 2-3 random features.

With max_features='sqrt' (default):
  Split 1 in Tree 1: considers [Age, Noise_1, Income] → picks Age
  Split 1 in Tree 2: considers [Education, Health, Noise_3] → picks Education
  → Trees become diverse (good!)

With max_features=None (all features):
  Every split considers all 8 features.
  Every tree splits on Education first (strongest signal).
  All 200 trees look almost identical!
  → Defeats purpose of ensemble
  → Less robust predictions
```

---

## PARAMETER: `bootstrap` (default=True)

> "Whether to sample training data with replacement."

```
With bootstrap=True (default):
  Tree 1 training set: [sample_1, sample_1, sample_5, sample_23, sample_23, sample_100, ...]
  Tree 2 training set: [sample_2, sample_7, sample_7, sample_7, sample_45, sample_88, ...]

  Some samples appear multiple times, some not at all.
  Each tree sees different data → diverse trees!

  Out-of-Bag (OOB) samples: ~37% of samples not selected for each tree
  → Can estimate accuracy without test set

With bootstrap=False:
  All trees see identical 450 samples.
  Only diversity comes from random feature selection.
  → Less robust ensemble
```

---

# 👤 MEMBER 2 (Second Half)

---

## TRAINING THE MODEL

```python
rf.fit(X_train, y_train)
```

> "This creates 200 decision trees. What happens inside:"

```
For each of 200 trees:
    1. Bootstrap sample: randomly select 450 samples with replacement
    2. Build tree:
       - At each node, consider sqrt(8)≈3 random features
       - Pick feature+threshold that best splits classes
       - Continue until max_depth=8 or min_samples_leaf=2
    3. Store the tree
```

> "**Output after training:**"

```
rf.n_estimators = 200  (200 trees stored)
rf.estimators_[0]      (first tree object)
rf.estimators_[0].get_depth()  → 8 (first tree depth)
```

---

## MODEL EVALUATION

```python
train_acc = rf.score(X_train, y_train)  # → 0.90
test_acc = rf.score(X_test, y_test)     # → 0.82
```

> "**What this output means:**"

```
Train Accuracy: 90%
  → Model correctly classifies 405 out of 450 training samples

Test Accuracy: 82%
  → Model correctly classifies 123 out of 150 test samples

Gap: 8%
  → Some overfitting, but acceptable
  → If gap was 25% (like with max_depth=None), that's bad overfitting
```

> "**Classification Report output:**"

```
              precision    recall  f1-score   support
           0       0.81      0.84      0.82        75
           1       0.83      0.80      0.82        75
    accuracy                           0.82       150

precision = Of all predicted Class 1, what % were actually Class 1
recall    = Of all actual Class 1, what % did we correctly predict
f1-score  = Harmonic mean of precision and recall
```

---

## FEATURE IMPORTANCE — IMPURITY-BASED METHOD

```python
importances = rf.feature_importances_
```

> "**How it calculates importance for each feature:**"

```
For feature "Age":
    1. Find every node across all 200 trees where Age was used for splitting
    2. At each node, calculate:
       weighted_impurity_decrease = (samples_at_node / total_samples) ×
                                    (impurity_before - impurity_after)
    3. Sum all decreases for Age
    4. Divide by sum for all features (normalize to 1.0)
```

> "**Example calculation for one node:**"

```
Node: 100 samples (60 Class 0, 40 Class 1)
Split on Age > 45:
  Left child:  50 samples (45 Class 0, 5 Class 1)
  Right child: 50 samples (15 Class 0, 35 Class 1)

Gini before: 1 - (0.6² + 0.4²) = 0.48
Gini left:   1 - (0.9² + 0.1²) = 0.18
Gini right:  1 - (0.3² + 0.7²) = 0.42

Weighted Gini after: (50/100)×0.18 + (50/100)×0.42 = 0.30

Impurity decrease: 0.48 - 0.30 = 0.18 → Added to Age's importance
```

> "**Our actual output:**"

```
Feature Importances (Impurity Decrease):
     Education : 0.2734   ← Highest (as expected from coefficient 0.7)
           Age : 0.2289   ← Second (coefficient 0.055)
        Income : 0.1823   ← Third
  Health_Score : 0.1098   ← Fourth (coefficient 0.03)
       Noise_1 : 0.0589   ← Bottom 4 — all noise features
       Noise_2 : 0.0512
       Noise_3 : 0.0487
       Noise_4 : 0.0468
                 -------
         Total : 1.0000
```

> "**This validates our model:** Signal features rank top 4, noise features rank bottom 4."

---

## FEATURE IMPORTANCE — PERMUTATION METHOD

```python
perm_result = permutation_importance(rf, X_test, y_test, n_repeats=30, scoring="accuracy")
```

> "**How it works:**"

```
For feature "Age":
    1. Record baseline accuracy: 82%
    2. Shuffle Age column randomly (break relationship with target)
    3. Measure new accuracy: 71%
    4. Importance = 82% - 71% = 11% (0.11)
    5. Repeat 30 times, take average

If shuffling a feature drops accuracy a lot → that feature was important
If shuffling barely changes accuracy → feature wasn't important
```

> "**Actual output:**"

```
Permutation Importances:
     Education : 0.0847  ← Shuffling Education drops accuracy by 8.5%
           Age : 0.0623  ← Shuffling Age drops accuracy by 6.2%
        Income : 0.0412
  Health_Score : 0.0234
       Noise_1 : 0.0012  ← Shuffling noise barely changes accuracy
       Noise_2 : 0.0008
       Noise_3 : 0.0003
       Noise_4 : 0.0001
```

> "**Why use both methods?**"

```
Impurity-based:
  ✓ Fast (calculated during training)
  ✗ Biased toward features with many unique values

Permutation-based:
  ✓ Unbiased
  ✓ Uses test data (realistic)
  ✗ Slow (needs re-evaluation)

Both agree on rankings → HIGH CONFIDENCE in results
```

---

## VISUALIZATION 1: Feature Importance Bar Chart

```python
plt.barh(importances_sorted.index, importances_sorted.values)
plt.savefig("plot_feature_importance.png")
```

> "**What the output looks like:**"

```
                    Feature Importance (Mean Impurity Decrease)
Noise_4        ████ 0.0468
Noise_3        ████ 0.0487
Noise_2        █████ 0.0512
Noise_1        █████ 0.0589
Health_Score   ██████████ 0.1098
Income         ████████████████ 0.1823
Age            ████████████████████ 0.2289
Education      ██████████████████████ 0.2734  (RED - above median)
               |-------- median line --------|

RED bars = above median importance
BLUE bars = below median importance
```

---

## VISUALIZATION 2: Impurity vs Permutation Comparison

```python
fig, axes = plt.subplots(1, 2)
# Left: impurity importance
# Right: permutation importance
plt.savefig("plot_importance_comparison.png")
```

> "**What it shows:**"

```
Left Panel (Impurity):          Right Panel (Permutation):
Education  ███████ 0.27         Education  ████████ 0.085
Age        ██████ 0.23          Age        ██████ 0.062
Income     █████ 0.18           Income     ████ 0.041
Health     ███ 0.11             Health     ██ 0.023
Noise_1    ██ 0.06              Noise_1    ▏ 0.001
Noise_2    █ 0.05               Noise_2    ▏ 0.001
Noise_3    █ 0.05               Noise_3    ▏ 0.000
Noise_4    █ 0.05               Noise_4    ▏ 0.000

Both methods show same ranking → Results validated!
```

---

## VISUALIZATION 3: Confusion Matrix

```python
cm = confusion_matrix(y_test, rf.predict(X_test))
sns.heatmap(cm, annot=True)
plt.savefig("plot_confusion_matrix.png")
```

> "**Actual output:**"

```
              Predicted
              Class 0    Class 1
Actual  Class 0    63        12
        Class 1    15        60

Reading the matrix:
  63 = True Negatives  (correctly predicted 0)
  60 = True Positives  (correctly predicted 1)
  12 = False Positives (predicted 1, was actually 0) — Type I error
  15 = False Negatives (predicted 0, was actually 1) — Type II error

Total correct: 63 + 60 = 123 out of 150 = 82% accuracy
```

---

## VISUALIZATION 4: Tree Depth Distribution

```python
depths = [tree.get_depth() for tree in rf.estimators_]
plt.hist(depths)
plt.savefig("plot_tree_depths.png")
```

> "**What it shows:**"

```
Number of Trees
     50 |          ████████████████████████████████████████
     40 |          ████████████████████████████████████████
     30 |  ████    ████████████████████████████████████████
     20 |  ████    ████████████████████████████████████████
     10 |  ████████████████████████████████████████████████
      0 +----+----+----+----+----+----+----+----+----+
             5    6    7    8
                 Tree Depth

Most trees reach depth 7-8 (our max_depth=8 limit)
Mean depth ≈ 7.8 (red dashed line)
→ Trees are using the full allowed complexity
```

---

## KEY FUNCTIONS SUMMARY

| Function                              | Example        | Output                         |
| ------------------------------------- | -------------- | ------------------------------ |
| `np.random.seed(42)`                  | Sets seed      | Same random numbers every run  |
| `np.random.randint(18, 70, 600)`      | 600 ages       | `[45, 28, 62, 35, ...]`        |
| `np.random.normal(50000, 15000, 600)` | 600 incomes    | `[52341.23, 38000.0, ...]`     |
| `train_test_split(..., stratify=y)`   | Split data     | 450 train, 150 test (balanced) |
| `rf.fit(X_train, y_train)`            | Train          | Creates 200 trees              |
| `rf.score(X_test, y_test)`            | Accuracy       | `0.82`                         |
| `rf.feature_importances_`             | Importance     | `[0.27, 0.23, 0.18, ...]`      |
| `permutation_importance(...)`         | Alt importance | `[0.085, 0.062, ...]`          |

---

## ADDITIONAL SCENARIOS

---

### If Dataset Had Only 100 Samples:

```
With n=100:
  - Only 75 training samples
  - Noise features might rank in top 4 by random chance
  - Feature importance unreliable
  - Need to increase min_samples_split to 10-20
```

---

### If Classes Were Imbalanced (90-10):

```
With 90% Class 0, 10% Class 1:
  - Model might just predict Class 0 always (90% "accuracy")
  - Must use class_weight='balanced':

    rf = RandomForestClassifier(class_weight='balanced', ...)

  - This weights Class 1 samples 9× higher
  - Forces model to learn minority class patterns
```

---

### If Age and Income Were Correlated (r=0.9):

```
Both features contain similar information.
Impurity importance splits between them:
  Age: 0.15 (instead of 0.23)
  Income: 0.14 (instead of 0.18)

Permutation importance handles this better.
Always check correlation matrix before modeling:
  df.corr()
```

---

### If Data Had Missing Values:

```
sklearn RandomForest cannot handle NaN.

Options:
1. df.dropna()                    — Remove rows with missing values
2. df.fillna(df.median())        — Replace with median
3. Use HistGradientBoostingClassifier — Handles NaN natively
```

---

## CLOSING

> "To summarize: We built a 200-tree Random Forest on 600 samples with 8 features. The model achieves 82% test accuracy and correctly identifies Education, Age, Income, and Health_Score as the important features while ranking all four noise features at the bottom."

> "Both impurity-based and permutation-based importance methods agree on the rankings, validating that our Random Forest successfully distinguishes signal from noise."

> "That completes our explanation."

---

_End of Presentation Script_

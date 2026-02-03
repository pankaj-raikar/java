# Breast Cancer Classification Using Random Forest
## Lab Evaluation Script (Detailed Version)

---

# STRUCTURE

**Member 1:** First Half (Introduction → Algorithm)  
**Member 2:** Second Half (Training → Results)

**Total Time:** ~15-18 minutes

---

# ═══════════════════════════════════════════════════════════════
# MEMBER 1 - FIRST HALF
# ═══════════════════════════════════════════════════════════════

---

## **What We Built**

"We built a **breast cancer classification model** using the **Random Forest algorithm**. It takes 30 cell measurements as input and predicts whether a tumor is malignant (cancerous) or benign (non-cancerous). Our model achieved **95.61% accuracy** on test data — correctly classifying 109 out of 114 samples."

---

## **Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
```

### **What each library does:**

| Library | What it does | Example in our code |
|---------|--------------|---------------------|
| **NumPy** | Handles numerical arrays and computations | Works behind the scenes when sklearn processes data |
| **Pandas** | Creates DataFrames — table-like structures with rows/columns | `X = pd.DataFrame(data.data, columns=data.feature_names)` — converts raw data into a labeled table |
| **Matplotlib** | Creates visualizations | `plt.show()` displays our confusion matrix and bar charts |
| **sklearn.datasets** | Provides built-in datasets | `load_breast_cancer()` loads the 569-sample dataset |
| **sklearn.model_selection** | Data splitting utilities | `train_test_split()` divides data into training and testing sets |
| **sklearn.ensemble** | Ensemble learning models | `RandomForestClassifier` — our main algorithm |
| **sklearn.metrics** | Evaluation tools | `accuracy_score`, `classification_report` measure model performance |

**If we didn't use Pandas:**  
We'd have raw numpy arrays without column names. Instead of `X['worst area']`, we'd need to remember that column 23 is worst area. Pandas makes data readable and easier to work with.

---

## **Dataset**

```python
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
```

### **Line-by-line explanation:**

**`data = load_breast_cancer()`**
- Returns a `Bunch` object (like a dictionary) containing:
  - `data.data` — 569×30 array of feature values
  - `data.target` — 569 labels (0=malignant, 1=benign)
  - `data.feature_names` — names of all 30 features
  - `data.target_names` — ['malignant', 'benign']
  - `data.DESCR` — full dataset description

**`X = pd.DataFrame(data.data, columns=data.feature_names)`**
- Converts the raw numpy array into a Pandas DataFrame
- `columns=data.feature_names` assigns meaningful column names
- Result: A table where each row is a sample, each column is a feature

**`y = data.target`**
- The target variable — what we want to predict
- Array of 569 values: 0s (malignant) and 1s (benign)

### **Dataset specifications:**

| Property | Value | Meaning |
|----------|-------|---------|
| Samples | 569 | 569 tumor samples |
| Features | 30 | 30 measurements per sample |
| Malignant (0) | 212 | 37% of samples are cancerous |
| Benign (1) | 357 | 63% of samples are non-cancerous |

### **The 30 features explained:**

There are 10 base features, each measured 3 ways:
- **Mean** — average value across the cell nucleus
- **SE (Standard Error)** — variability in the measurement
- **Worst** — largest (most extreme) value found

**Example: For "radius"**
- `mean radius` = average distance from center to edge
- `radius error` = how much the radius varies
- `worst radius` = the largest radius measurement

**Why "worst" features matter for cancer:**  
Cancer cells are irregular. The "worst" values capture the most abnormal cell characteristics — often the best indicators of malignancy.

```python
print("Feature matrix shape:", X.shape)   # (569, 30)
print("Target vector shape:", y.shape)    # (569,)
X.head()  # Shows first 5 rows of the DataFrame
```

**`X.shape` returns `(569, 30)`** meaning 569 rows (samples) and 30 columns (features).

---

## **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### **What this function returns:**

| Variable | Content | Size |
|----------|---------|------|
| `X_train` | Training features | 455 samples × 30 features |
| `X_test` | Testing features | 114 samples × 30 features |
| `y_train` | Training labels | 455 labels |
| `y_test` | Testing labels | 114 labels |

### **Parameter-by-parameter explanation:**

---

**`test_size=0.2`**

*What it does:* Reserves 20% of data for testing, 80% for training.

*In our case:*
- Training: 455 samples (80%)
- Testing: 114 samples (20%)

*What if we change it?*
| Value | Effect |
|-------|--------|
| `test_size=0.1` | 90% train, 10% test. More training data but less reliable accuracy estimate |
| `test_size=0.3` | 70% train, 30% test. More reliable testing but model learns from less data |
| `test_size=0.5` | 50-50 split. Good testing but may not train well with only half the data |

*Rule of thumb:* 0.2 (20%) is a common choice — enough data to train and enough to test.

---

**`random_state=42`**

*What it does:* Sets a seed for the random number generator. Ensures the same split every time we run the code.

*In our case:* Every time we run this code, the exact same 114 samples go to testing and 455 to training.

*What if we change it?*
| Value | Effect |
|-------|--------|
| `random_state=42` | Always same split. Reproducible results. |
| `random_state=0` | Different fixed split (still reproducible) |
| `random_state=None` | Different split every run. Results vary each time. |

*Why 42?* Any integer works. 42 is commonly used (reference to "Hitchhiker's Guide to the Galaxy"). The specific number doesn't matter — what matters is using the same number for reproducibility.

*Why reproducibility matters:* If you're tuning parameters, you want the only change to be your parameter — not the data split.

---

**`stratify=y`**

*What it does:* Ensures training and testing sets have the same class proportions as the original data.

*In our case:*
- Original: 37% malignant, 63% benign
- With stratify: Both train and test sets are ~37% malignant, ~63% benign
- Without stratify: Could randomly get 50% malignant in one set, 25% in another

*What if we remove it?*
| Setting | Effect |
|---------|--------|
| `stratify=y` | Proportions preserved. Fair evaluation. |
| No stratify | Random proportions. Might put mostly benign in training, mostly malignant in testing. Could make model seem worse/better than it is. |

*Example problem without stratify:* If training set gets 90% benign samples, model learns "just predict benign" — then fails on test set with different proportions.

---

# ═══════════════════════════════════════════════════════════════
# 🌳 UNDERSTANDING DECISION TREES (IMPORTANT!)
# ═══════════════════════════════════════════════════════════════

*This is the core concept. Make sure you understand this before moving to Random Forest.*

---

## **What is a Decision Tree?**

A decision tree is like a **flowchart of yes/no questions** that leads to an answer.

### **Simple Real-Life Example: Should I take an umbrella?**

```
                 Is it raining?
                /              \
              YES               NO
               |                 |
         Take umbrella    Is forecast rainy?
                          /              \
                        YES               NO
                         |                 |
                  Take umbrella    Don't take umbrella
```

This is a decision tree! You start at the top and follow the path based on answers until you reach a decision.

---

## **Decision Tree for Cancer Classification (Our Program)**

In our program, a decision tree asks questions about cell measurements:

```
                    Is worst area ≤ 696?
                   /                      \
                 YES                       NO
                  |                         |
      Is worst concave points ≤ 0.135?    Is worst concave points ≤ 0.091?
         /                \                  /                    \
       YES                NO               YES                    NO
        |                  |                |                      |
      BENIGN          Is mean texture     BENIGN               MALIGNANT
                         ≤ 19.5?
                        /      \
                      YES       NO
                       |         |
                    BENIGN    MALIGNANT
```

### **How to read this:**

**Example 1:** A tumor sample has:
- worst area = 500 (≤ 696? YES → go left)
- worst concave points = 0.08 (≤ 0.135? YES → go left)
- **Result: BENIGN**

**Example 2:** A tumor sample has:
- worst area = 900 (≤ 696? NO → go right)
- worst concave points = 0.15 (≤ 0.091? NO → go right)
- **Result: MALIGNANT**

---

## **How the Tree is Built (Training)**

When we call `rf_model.fit()`, each tree learns by finding the best questions to ask.

**Step 1: Start with all 455 training samples at the root**

```
                    [455 samples]
                 [168 malignant, 287 benign]
```

**Step 2: Find the best question to split**

The algorithm tries every possible question:
- "Is mean radius ≤ 10?" 
- "Is worst area ≤ 500?"
- "Is worst area ≤ 600?"
- ... thousands of possibilities

For each question, it calculates how well it separates malignant from benign.

**Step 3: Pick the best split**

Let's say "Is worst area ≤ 696?" is best:

```
                    Is worst area ≤ 696?
                   /                      \
          [280 samples]              [175 samples]
       [35 malignant, 245 benign]   [133 malignant, 42 benign]
```

- Left side: Mostly benign (245 vs 35) — good!
- Right side: Mostly malignant (133 vs 42) — good!

**Step 4: Repeat for each branch**

Keep splitting until:
- Leaves are "pure" (all same class), OR
- A stopping condition is met (like `min_samples_split`)

---

## **One Tree vs. Many Trees (Random Forest)**

### **The Problem with One Tree:**

A single decision tree:
- Can memorize the training data (overfitting)
- Makes mistakes if the training data has noise
- Different random splits of data → very different trees

### **The Solution: Random Forest = 100 Trees**

Instead of trusting one tree, we build **100 different trees** and let them **vote**.

```
           Sample to classify: [worst area=750, concave points=0.12, ...]
                                        |
            ┌──────────────────────────────────────────────────┐
            |                                                  |
      ┌─────┴─────┐    ┌─────┴─────┐          ┌─────┴─────┐
      │   Tree 1  │    │   Tree 2  │   ...    │  Tree 100 │
      │   Vote:   │    │   Vote:   │          │   Vote:   │
      │  BENIGN   │    │ MALIGNANT │          │  BENIGN   │
      └───────────┘    └───────────┘          └───────────┘
            |                |                       |
            └────────────────┴───────────────────────┘
                              |
                        FINAL VOTE
                    65 Benign, 35 Malignant
                              |
                    PREDICTION: BENIGN
```

### **Why 100 Trees are Better Than 1:**

| Aspect | 1 Tree | 100 Trees (Random Forest) |
|--------|--------|---------------------------|
| Overfitting | High risk | Low risk (averaging reduces it) |
| Stability | Sensitive to small data changes | Stable predictions |
| Accuracy | Lower | Higher |
| Individual errors | Hurt the result | Averaged out by other trees |

**Analogy:** Asking 1 doctor vs. asking 100 doctors for a diagnosis. If 65 doctors say benign and 35 say malignant, you go with benign.

---

## **What Makes Each Tree "Random"?**

Two sources of randomness:

### **1. Bootstrap Sampling (Random Samples)**

Each tree gets a different random subset of the 455 training samples.

- Tree 1 might get samples: [1, 3, 3, 5, 7, 8, 8, ...] (with replacement)
- Tree 2 might get samples: [2, 4, 4, 6, 9, 12, ...] (different subset)

*With replacement* means the same sample can appear multiple times.

### **2. Random Feature Selection (Random Questions)**

At each split, only a random subset of the 30 features is considered.

- Tree 1 at node A: Considers [worst area, mean radius, symmetry, ...] (5-6 random features)
- Tree 1 at node B: Considers [texture, smoothness, concavity, ...] (different 5-6 features)

**Why randomness helps:**
- Trees become different from each other
- No single feature dominates all trees
- Errors are independent → averaging works better

---

# ═══════════════════════════════════════════════════════════════
# BACK TO THE CODE
# ═══════════════════════════════════════════════════════════════

---

## **Random Forest Model**

```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)
```

Now you know:
- **100 trees** will be built (`n_estimators=100`)
- Each tree asks yes/no questions about features
- Each tree grows fully until leaves are pure (`max_depth=None`)
- All trees vote to make the final prediction

### **Parameter-by-parameter explanation:**

---

**`n_estimators=100`**

*What it does:* Number of decision trees in the forest.

*In our case:* 100 trees vote on each prediction.

*What if we change it?*
| Value | Effect |
|-------|--------|
| `n_estimators=10` | Faster training, but less accurate. Fewer trees to vote. |
| `n_estimators=100` | Good balance of speed and accuracy (our choice) |
| `n_estimators=500` | Potentially more accurate, but 5× slower to train |
| `n_estimators=1000` | Diminishing returns — marginally better but much slower |

*Example:* Imagine 10 doctors vs 100 doctors voting. 100 opinions average out better, but 1000 doctors takes too long for small extra benefit.

---

**`max_depth=None`**

*What it does:* Maximum depth of each tree. `None` means trees grow until all leaves are pure.

*What is depth?* The number of levels of questions.
```
Level 0 (root):     Is worst area ≤ 696?
                   /                    \
Level 1:      Question                Question
              /      \                /      \
Level 2:   Question  Answer      Answer    Question
             ...                              ...
```

*In our case:* Trees can be depth 10, 15, 20 — whatever is needed.

*What if we change it?*
| Value | Effect |
|-------|--------|
| `max_depth=None` | Full trees. Maximum learning. |
| `max_depth=3` | Only 3 levels of questions. Simple trees. Might miss complex patterns. |
| `max_depth=5` | Moderate depth. |
| `max_depth=10` | Deeper, but still limited. |

*Visualization note:* In our code, we use `max_depth=3` for visualization because full trees are too complex to display.

---

**`min_samples_split=2`**

*What it does:* Minimum samples needed to split a node further.

*In our case:* A node needs at least 2 samples to try another question.

*Example:*
- Node has 100 samples → Can split
- Node has 5 samples → Can split
- Node has 1 sample → Becomes a leaf (no more splitting)

*What if we change it?*
| Value | Effect |
|-------|--------|
| `min_samples_split=2` | Very deep trees. Splits until almost pure. |
| `min_samples_split=10` | Node needs 10+ samples to split. Shallower trees. |
| `min_samples_split=50` | Very shallow trees. Only major patterns. |

---

**`random_state=42`**

*What it does:* Seed for all randomness (bootstrap sampling, feature selection).

*In our case:* Same 100 trees are built every run.

---

*[Handover to Member 2]*

"That covers the setup and what decision trees/forests are. My partner will now explain the training process and results."

---

# ═══════════════════════════════════════════════════════════════
# MEMBER 2 - SECOND HALF
# ═══════════════════════════════════════════════════════════════

---

## **Training the Model**

```python
rf_model.fit(X_train, y_train)
```

### **What `fit()` does — Step by Step:**

**Step 1: For each of the 100 trees, create a bootstrap sample**

Tree 1 gets: 455 random samples from training data (with replacement)
Tree 2 gets: 455 different random samples
... (100 trees total)

**Step 2: Build each tree**

For each tree, starting at the root:
1. Take all samples at this node
2. Randomly select ~5 features (√30 ≈ 5)
3. Find the best feature and threshold to split on
4. Split into two child nodes
5. Repeat for each child until done

**Step 3: Store all trees**

All 100 trees are saved in `rf_model.estimators_` for later predictions.

### **Gini Impurity — How "best" split is chosen:**

At each node, we want to separate malignant from benign as cleanly as possible.

**Formula:** `Gini = 1 - (p₀² + p₁²)`

Where:
- p₀ = proportion of malignant samples in the node
- p₁ = proportion of benign samples in the node

**Examples:**

| Node contents | p₀ | p₁ | Gini | Meaning |
|---------------|----|----|------|---------|
| 100 malignant, 0 benign | 1.0 | 0.0 | 1 - (1 + 0) = **0** | Pure! All same class. |
| 0 malignant, 100 benign | 0.0 | 1.0 | 1 - (0 + 1) = **0** | Pure! |
| 50 malignant, 50 benign | 0.5 | 0.5 | 1 - (0.25 + 0.25) = **0.5** | Maximum impurity (50-50) |
| 30 malignant, 70 benign | 0.3 | 0.7 | 1 - (0.09 + 0.49) = **0.42** | Somewhat impure |

**Lower Gini = better split**

**Example of choosing a split:**

Node has: 100 samples (40 malignant, 60 benign), Gini = 0.48

Try split: "Is worst area ≤ 500?"
- Left child: 10 malignant, 55 benign → Gini = 0.27
- Right child: 30 malignant, 5 benign → Gini = 0.24

Average Gini after split = ~0.26 (much better than 0.48!)

The algorithm picks the split that reduces Gini the most.

---

## **Making Predictions**

```python
y_pred = rf_model.predict(X_test)
```

### **What happens for ONE test sample:**

Let's say test sample #47 has these values:
- worst area = 620
- worst concave points = 0.14
- mean radius = 15.2
- ... (30 features total)

**Each tree makes its own prediction:**

**Tree 1:**
```
Is worst area ≤ 696? → 620 ≤ 696? YES → Go left
Is worst concave points ≤ 0.135? → 0.14 ≤ 0.135? NO → Go right
Is mean texture ≤ 19.5? → 18 ≤ 19.5? YES → Go left
→ BENIGN
```

**Tree 2:**
```
Is mean radius ≤ 14.5? → 15.2 ≤ 14.5? NO → Go right
Is worst perimeter ≤ 100? → 95 ≤ 100? YES → Go left
→ BENIGN
```

**Tree 3:**
```
Is worst concave points ≤ 0.10? → 0.14 ≤ 0.10? NO → Go right
Is worst area ≤ 800? → 620 ≤ 800? YES → Go left
→ MALIGNANT
```

... (97 more trees vote)

**Final tally:**
- 68 trees say BENIGN
- 32 trees say MALIGNANT
- **Majority wins → BENIGN**

This is repeated for all 114 test samples.

---

## **Accuracy Score**

```python
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)  # Output: 0.9561 (95.61%)
```

### **What `accuracy_score()` does:**

**Formula:** `Accuracy = Correct Predictions / Total Predictions`

*In our case:*
- Total test samples: 114
- Correct predictions: 109
- Accuracy: 109/114 = **0.9561 = 95.61%**

---

## **Classification Report**

```python
print(classification_report(y_test, y_pred))
```

**Output:**
```
              precision    recall  f1-score   support
           0       0.95      0.93      0.94        42
           1       0.96      0.97      0.97        72
    accuracy                           0.96       114
```

### **Each metric with real numbers from our model:**

---

**Precision** = TP / (TP + FP) — "How accurate are our positive predictions?"

For malignant (class 0):
- We predicted 41 samples as malignant
- 39 were actually malignant (TP), 2 were benign (FP)
- Precision = 39/(39+2) = **0.95 (95%)**

---

**Recall** = TP / (TP + FN) — "What % of actual positives did we find?"

For malignant (class 0):
- There were 42 actual malignant samples
- We found 39 (TP), missed 3 (FN)
- Recall = 39/(39+3) = **0.93 (93%)**

---

**F1-Score** = 2 × (P × R) / (P + R) — Harmonic mean of precision and recall

For malignant: 2 × (0.95 × 0.93) / (0.95 + 0.93) = **0.94**

---

**Support** = Number of actual samples in each class
- Malignant: 42
- Benign: 72
- Total: 114

---

## **Confusion Matrix**

```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.show()
```

### **Our confusion matrix:**

|  | **Predicted Malignant** | **Predicted Benign** |
|--|------------------------|---------------------|
| **Actual Malignant** | 39 (TP) ✓ | 3 (FN) ✗ |
| **Actual Benign** | 2 (FP) ✗ | 70 (TN) ✓ |

**Reading it:**
- **39 True Positives:** Cancer → Predicted cancer ✓
- **70 True Negatives:** No cancer → Predicted no cancer ✓
- **3 False Negatives:** Cancer → Predicted no cancer ✗ (MISSED!)
- **2 False Positives:** No cancer → Predicted cancer ✗ (false alarm)

**Why 3 False Negatives are concerning:**
In real life, these 3 patients would be told "no cancer" when they actually have cancer. They might not get treatment.

---

## **Feature Importance**

```python
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feature_importance_df.head(10)
```

### **What this tells us:**

Which features are most useful for making predictions.

**How importance is calculated:**
1. Every time a feature is used to split a node, the Gini reduction is recorded
2. Sum across all splits in all 100 trees
3. Normalize so all values sum to 1.0

### **Our top features:**

| Feature | Importance | Why it matters |
|---------|------------|----------------|
| worst area | 14.0% | Largest cell size — cancer cells are bigger |
| worst concave points | 13.0% | Irregular cell shapes |
| worst radius | 9.8% | Cell size |
| mean concave points | 9.1% | Shape irregularity |
| worst perimeter | 7.2% | Boundary length |

**Key insight:** "Worst" features (extreme values) are most predictive because cancer cells have abnormal extremes.

---

## **Decision Tree Visualization**

```python
from sklearn.tree import plot_tree

tree = rf_model.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3
)
plt.title("One Decision Tree from Random Forest (Depth = 3)")
plt.show()
```

### **What `estimators_[0]` means:**

- `estimators_` = list of all 100 trees in our forest
- `[0]` = first tree (index 0)
- We visualize just 1 tree because showing 100 would be impractical

### **Reading a node:**

```
worst perimeter <= 106.1
gini = 0.468
samples = 455
value = [168, 287]
class = benign
```

- **Line 1:** The question asked at this node
- **gini:** Impurity (0 = pure, 0.5 = 50-50 split)
- **samples:** Training samples at this node
- **value:** [malignant count, benign count]
- **class:** Current majority prediction

---

## **Saving Results**

```python
feature_importance_df.to_csv("feature_importance_results.csv", index=False)
```

Saves feature importance to a CSV file.

---

## **Summary**

| What | Value |
|------|-------|
| Algorithm | Random Forest (100 decision trees voting) |
| Dataset | 569 samples, 30 features |
| Training | 455 samples (80%) |
| Testing | 114 samples (20%) |
| Accuracy | 95.61% (109/114 correct) |
| Missed cancers | 3 (False Negatives) |
| False alarms | 2 (False Positives) |
| Best predictor | worst area (14%) |

---

# ═══════════════════════════════════════════════════════════════
# QUICK Q&A REFERENCE
# ═══════════════════════════════════════════════════════════════

**"What is a decision tree?"**
A flowchart of yes/no questions that leads to a prediction. Each question splits the data based on a feature value.

**"What is Random Forest?"**
100 decision trees built with random samples and random features. They vote to make predictions.

**"Why use 100 trees instead of 1?"**
One tree can overfit. 100 trees average out errors and give more stable, accurate predictions.

**"What does fit() do?"**
Builds all 100 trees by learning the best questions (splits) from training data.

**"What does predict() do?"**
Passes a sample through all 100 trees, counts votes, returns the majority class.

**"What is Gini impurity?"**
Measures how mixed a node is. Gini=0 means pure (all same class). Gini=0.5 means 50-50 split.

**"Why are 'worst' features most important?"**
Cancer cells have abnormal extremes. The "worst" (largest) measurements capture these abnormalities.

**"What happens if n_estimators=10?"**
Only 10 trees vote instead of 100. Faster but less accurate.

**"What does random_state do?"**
Sets a seed so the same random choices are made every run. Makes results reproducible.

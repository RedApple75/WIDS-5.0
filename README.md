# Plant Disease Classification Analysis

This project analyzes a dataset of plant leaf images to classify various species and disease conditions. The workflow progresses from Exploratory Data Analysis (EDA) to implementing shallow machine learning models and hierarchical classification strategies.

## 1. Dataset Insights (EDA)

The dataset consists of images representing different plant species and their health status.

**Key Statistics:**
* **Total Images:** 54,305
* **Total Classes:** 38 (Specific Plant + Disease combinations)
* **Input Resolution:** Resized to 64x64 pixels
* **Color Channels:** 3 (RGB)

**Class Distribution:**
The dataset is highly imbalanced. The majority class (Orange Haunglongbing) contains significantly more samples than the minority class (Potato healthy).



*Figure 1: Distribution of images across the 38 classes, highlighting significant imbalance.*

## 2. Data Representation

Before feeding images into shallow models, we flatten the 2D image matrices into 1D vectors.

For an image with height $H$, width $W$, and channels $C$, the input vector $x$ is:

$$
x \in \mathbb{R}^{H \times W \times C}
$$

With our dimensions ($64 \times 64 \times 3$):

$$
x \in \mathbb{R}^{12288}
$$

## 3. Model Architectures

### Support Vector Machine (SVM)

We use an SVM to find a linear boundary (hyperplane) that separates the classes.



**The Hyperplane:**
For a binary case, the decision boundary is defined by weights $w$ and bias $b$:

$$
w^T x + b = 0
$$

**Optimization (Hinge Loss):**
The goal is to maximize the margin between classes while minimizing errors. We minimize the cost function $J(w, b)$:

$$
\min_{w, b} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w^T x_i + b))
$$

* $||w||^2$: Regularization term (keeps weights small).
* $C$: Penalty parameter (controls trade off between smooth boundary and classifying training points correctly).
* $\max(0, ...)$: The Hinge Loss, which is zero if the point is correctly classified outside the margin.

### Random Forest Classifier

The Random Forest aggregates predictions from multiple Decision Trees to reduce overfitting and capture non linear patterns.



**Splitting Criterion:**
At each node, the tree splits data to maximize purity. We use Gini Impurity. For a node $t$ with $K$ classes, the impurity is:

$$
Gini(t) = 1 - \sum_{k=1}^{K} (p_{k})^2
$$

* $p_k$: The probability of a randomly chosen element belonging to class $k$.

The algorithm selects the split that maximizes the Information Gain (reduction in impurity):

$$
Gain = Gini(parent) - \sum_{child} \frac{N_{child}}{N_{parent}} Gini(child)
$$

**Ensemble Prediction:**
For a forest of $B$ trees, the final class $\hat{y}$ is the majority vote:

$$
\hat{y} = \text{mode} \{ T_1(x), T_2(x), ..., T_B(x) \}
$$

### Dual Label Structure (Hierarchical)

Standard classification treats "Tomato___Bacterial_spot" as a single label. The Dual Model splits this into two tasks:

1.  **Plant Prediction:** Is it a Tomato, Potato, etc.?
2.  **Disease Prediction:** Is it Bacterial spot, Healthy, etc.?

This models the conditional probability:

$$
P(\text{Class}) \approx P(\text{Plant}) \cdot P(\text{Disease} | \text{Plant})
$$

This approach helps isolate whether the model fails at recognizing the leaf shape (Plant error) or the texture of the spot (Disease error).

## 4. Evaluation Metrics

Given the class imbalance, Accuracy is not sufficient. We rely on Precision, Recall, and F1 Score.

**F1 Score:**
The harmonic mean of Precision and Recall:

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

**Macro Average:**
To treat all classes equally regardless of their size (important for our imbalanced dataset):

$$
\text{Macro } F1 = \frac{1}{N} \sum_{i=1}^{N} F1_i
$$
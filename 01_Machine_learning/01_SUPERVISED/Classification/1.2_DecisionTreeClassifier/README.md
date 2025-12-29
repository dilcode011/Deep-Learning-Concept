# Decision Tree Classifier

## 1. The Concept
A Decision Tree is a flowchart-like tree structure where:
* **Internal Node:** Represents a feature (or attribute) to check (e.g., "Is Age > 18?").
* **Branch:** Represents a decision rule (e.g., "Yes" or "No").
* **Leaf Node:** Represents the outcome (the final class label).

The goal is to split the data into groups that are as pure as possible (containing only one type of class).



## 2. The Math Behind Splitting
How does the tree decide which feature to split on first? It uses metrics to measure "impurity" or "disorder."

### A. Gini Impurity (Default in Scikit-Learn)
Measures the probability of incorrectly classifying a randomly chosen element.
$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
* **0.0**: Perfectly pure (all elements belong to one class).
* **0.5**: Maximum impurity (elements are evenly distributed across classes).

### B. Entropy (Information Gain)
Measures the disorder or uncertainty in the data.
$$Entropy = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
* The algorithm calculates the **Information Gain** (Reduction in Entropy) for every possible split and chooses the one that reduces entropy the most.

## 3. Pruning (Avoiding Overfitting)
Decision Trees are prone to **overfitting** (creating a massive tree that memorizes every single data point).
* **Pre-Pruning:** Setting limits *before* building (e.g., `max_depth=3`).
* **Post-Pruning:** Cutting back branches *after* building if they don't add much value.

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

### 📺 Videos
* [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
* [Visual Explanation of Decision Trees](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) (Interactive Website)
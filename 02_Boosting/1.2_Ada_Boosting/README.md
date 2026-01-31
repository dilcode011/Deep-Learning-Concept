# AdaBoost (Adaptive Boosting)

## 1. The Concept
AdaBoost combines many "weak learners" (usually Decision Stumps, which are trees with only one split) to create a strong learner.

### The "Adaptive" Mechanism
1.  **Round 1:** Train a model. Some points are classified correctly, others incorrectly.
2.  **Round 2:** **Increase the weight** of the incorrectly classified points. The next model is forced to focus on these "hard" examples.
3.  **Repeat:** This continues until the errors are minimized.



## 2. Key Differences
* **Tree Depth:** AdaBoost typically uses "Stumps" (depth=1). XGBoost/LightGBM use deeper trees.
* **Weighting:** AdaBoost weighs *data points*. Gradient Boosting fits *residuals* (errors).

## 3. Resources
* [Scikit-Learn: AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
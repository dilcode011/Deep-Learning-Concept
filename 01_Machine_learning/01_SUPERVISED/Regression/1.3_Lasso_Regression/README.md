# Lasso Regression (L1 Regularization)

## 1. The Concept
Lasso (Least Absolute Shrinkage and Selection Operator) is a regression analysis method that performs both **variable selection** and **regularization**.

Like Ridge, it shrinks coefficients to prevent overfitting. However, unlike Ridge, Lasso can shrink coefficients all the way to **zero**.

### The "Superpower": Feature Selection
If you have a dataset with 1,000 columns (features) but only 10 are actually important, Lasso will automatically turn the other 990 coefficients to 0. This makes the model sparse and easier to interpret.



## 2. The Math Behind It

### The Cost Function
We modify the Mean Squared Error (MSE) by adding the magnitude of coefficients as the penalty term.

$$J(\beta) = \underbrace{MSE}_{\text{Standard Error}} + \underbrace{\lambda \sum_{i=1}^{n} |\beta_i|}_{\text{L1 Penalty}}$$

* **$\lambda$ (Lambda/Alpha):** The tuning parameter.
* **$|\beta|$:** We take the *absolute value* of weights (L1 norm).

### Geometric Intuition
The constraint region for Lasso is a **diamond** (because of the absolute value), which has corners. The solution often hits a corner where one axis is 0.
The constraint region for Ridge is a **circle**, so the solution rarely hits exactly 0.



## 3. Lasso vs. Ridge Comparison
| Feature | Ridge (L2) | Lasso (L1) |
| :--- | :--- | :--- |
| **Penalty** | Square of coefficients ($\beta^2$) | Absolute value of coefficients ($|\beta|$) |
| **Zero Coefficients** | No (only close to zero) | **Yes** (Exact zero) |
| **Feature Selection** | No | **Yes** |
| **Best Used When** | All features are somewhat useful | Many features are useless/noise |

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: Lasso Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

### 📺 Videos
* [StatQuest: Lasso Regression Explained](https://www.youtube.com/watch?v=NGf0voTMlcs)
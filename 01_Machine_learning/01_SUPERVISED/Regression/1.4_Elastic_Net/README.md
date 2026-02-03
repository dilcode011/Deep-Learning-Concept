# Elastic Net Regression

## 1. The Concept
Elastic Net is a middle ground between **Ridge (L2)** and **Lasso (L1)**. It includes both penalty terms in the cost function.

It solves specific limitations of Lasso:
1.  **Correlated Features:** If two features are highly correlated (e.g., "Left Shoe Size" and "Right Shoe Size"), Lasso might randomly pick one and drop the other. Elastic Net is more likely to keep both.
2.  **Number of features > Number of samples:** In cases where $p > n$, Lasso behaves erratically. Elastic Net handles this better.

**The Strategy:** We penalize the model using *both* the absolute value of coefficients (L1) and the square of coefficients (L2).

### Why do we need it?
* **Ridge** is good at shrinking coefficients but cannot set them to 0 (no feature selection).
* **Lasso** is good at feature selection but struggles when features are **correlated** (it arbitrarily picks one and drops the others).
* **Elastic Net** groups correlated features together. If one is selected, the whole group is selected.



## 2. The Math Behind It

### The Cost Function
The cost function is a weighted combination of L1 and L2 penalties.

$$J(\beta) = MSE + r\lambda \sum_{i=1}^{n} |\beta_i| + \frac{1-r}{2}\lambda \sum_{i=1}^{n} \beta_i^2$$

* **$MSE$**: Mean Squared Error.
* **$\lambda$ (Alpha)**: Total penalty strength.
* **$r$ (L1 Ratio)**: The mix ratio.
    * If $r = 1$: It becomes pure **Lasso**.
    * If $r = 0$: It becomes pure **Ridge**.
    * If $0 < r < 1$: It is **Elastic Net**.

## 3. When to use which?
1.  **Linear Regression**: If you have lots of data and few features (low noise).
2.  **Ridge**: If you have many features with small effects (prevents overfitting).Handles multicollinearity well. |
3.  **Lasso**: If you have many features but only a few are important (sparse solution).
4.  **Elastic Net**: If you have many features and **some are correlated** (e.g., "square footage" and "number of rooms" usually move together).It is the safest bet. |

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: Elastic Net](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

### 📺 Videos
* [StatQuest: Elastic Net Explained](https://www.youtube.com/watch?v=1dKRdX9bfBc)
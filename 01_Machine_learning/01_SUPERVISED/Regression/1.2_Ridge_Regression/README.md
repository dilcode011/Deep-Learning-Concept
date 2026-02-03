# Ridge Regression (L2 Regularization)

## 1. The Concept
Ridge Regression is a regularized version of Linear Regression and it is also known as a Tikhonov regularization. It is used when a model is **overfitting** (memorizing the noise in the training data) or when there is **multicollinearity** (independent variables are highly correlated).

### Why do we need it?
In standard Linear Regression, the model tries to minimize the error completely, which sometimes leads to huge coefficients ($\beta$ values) that make the model unstable. Ridge adds a **penalty** to the cost function to keep these coefficients small.

> **Key Idea:** "It is better to have a slightly higher error on the training data if it means the model is simpler and generalizes better to new data."



## 2. The Math Behind It

### The Cost Function
We modify the standard Mean Squared Error (MSE) by adding a "Penalty Term".

$$J(\beta) = \underbrace{MSE}_{\text{Standard Error}} + \underbrace{\lambda \sum_{i=1}^{n} \beta_i^2}_{\text{L2 Penalty}}$$

* **$MSE$:** Tries to fit the data perfectly.
* **$\lambda$ (Lambda/Alpha):** The tuning parameter.
    * If $\lambda = 0$: It becomes standard Linear Regression.
    * If $\lambda \to \infty$: The coefficients approach zero (leads to underfitting).
* **$\beta^2$:** We square the weights (L2 norm), which forces them to be small but rarely exactly zero.



## 3. Ridge vs. Linear Regression
| Feature | Linear Regression | Ridge Regression |
| :--- | :--- | :--- |
| **Objective** | Minimize Error | Minimize Error + Keep Weights Small |
| **Overfitting** | Prone to overfitting | Reduces overfitting |
| **Coefficients** | Can be very large | Shrunk towards zero |
| **Use Case** | Simple, low-noise data | High-dimensional, noisy data |

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### 📺 Videos
* [StatQuest: Ridge Regression (L2) Explained](https://www.youtube.com/watch?v=Q81RR3yKn30) - *Highly Recommended*
* [Andrew Ng: Regularization](https://www.youtube.com/watch?v=u73PU6Qwl1I)
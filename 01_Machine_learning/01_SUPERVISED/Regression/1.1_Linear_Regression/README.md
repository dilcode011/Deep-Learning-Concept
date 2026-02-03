# Linear Regression

## 1. The Concept
Linear Regression is a supervised learning algorithm used for **regression tasks** (predicting a continuous value, like price, age, or temperature).

The goal is to find the "Best Fit Line" that passes through the data points in a way that minimizes the error between the actual values and the predicted values.



## 2. The Math Behind It

### The Hypothesis (Equation of the Line)
The relationship between the input ($x$) and output ($y$) is modeled as:
$$y = \beta_0 + \beta_1x + \epsilon$$

* $y$: Predicted value (Target)
* $x$: Input feature
* $\beta_1$: Slope (Coefficient) - How much $y$ changes for a unit change in $x$.
* $\beta_0$: Intercept (Bias) - Where the line crosses the y-axis.

### The Cost Function (MSE)
How do we know if our line is good? We calculate the **Mean Squared Error (MSE)**. We want to minimize this value.
$$MSE = \frac{1}{N} \sum_{i=1}^{n} (y_i - (mx_i + c))^2$$

### Optimization: Gradient Descent
To find the best $\beta_0$ and $\beta_1$, we use an iterative algorithm called Gradient Descent. It slowly tweaks the parameters to move "downhill" toward the lowest error.

## 3. Assumptions of Linear Regression
1.  **Linearity:** The relationship between X and Y is linear.
2.  **Independence:** Observations are independent of each other.
3.  **Homoscedasticity:** The variance of residuals is constant.
4.  **Normality:** The residuals (errors) follow a normal distribution.

## 4. Resources
### 📖 Documentation & Articles
* [Scikit-Learn Official Docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Towards Data Science: Linear Regression Explained](https://towardsdatascience.com/linear-regression-detailed-view-ea73175f6e86)

### 📺 Videos
* [Campusx: Linear Regression Main Ideas](https://youtu.be/UZPfbG0jNec)
* [Andrew Ng: Gradient Descent (Coursera)](https://www.youtube.com/watch?v=F6GSRDoB-Cg)



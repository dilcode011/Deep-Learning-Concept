# Logistic Regression

## 1. The Concept
Logistic Regression is a supervised learning algorithm used for **Classification** tasks. Unlike Linear Regression, which predicts a continuous number (like price), Logistic Regression predicts the **probability** that an instance belongs to a specific class.

The output is always between **0 and 1**.

### Use Cases
* **Spam Detection:** Is this email Spam (1) or Not Spam (0)?
* **Medical Diagnosis:** Is the tumor Malignant (1) or Benign (0)?
* **Credit Card Fraud:** Is this transaction Fraud (1) or Legit (0)?

## 2. The Math Behind It

### The Sigmoid Function (Logistic Function)
Linear Regression ($z = mx + b$) outputs values from $-\infty$ to $+\infty$. To map this to a probability between 0 and 1, we pass the result through the **Sigmoid Function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

* If $z$ is very positive, $\sigma(z) \approx 1$
* If $z$ is very negative, $\sigma(z) \approx 0$
* If $z = 0$, $\sigma(z) = 0.5$



### The Decision Boundary
Once we get a probability (e.g., 0.85), we apply a threshold (usually 0.5) to classify the data:
* $P(y=1) \ge 0.5 \rightarrow$ Class 1
* $P(y=1) < 0.5 \rightarrow$ Class 0

### Cost Function: Log Loss (Binary Cross Entropy)
We cannot use Mean Squared Error (MSE) here because the Sigmoid function makes the curve "wavy" (non-convex), making it hard to find the minimum error. Instead, we use **Log Loss**:

$$J(\theta) = - \frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

## 3. Resources
### 📖 Documentation
* [Scikit-Learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### 📺 Videos
* [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
* [Andrew Ng: Logistic Regression Model](https://www.youtube.com/watch?v=-la3q9d7AKQ)
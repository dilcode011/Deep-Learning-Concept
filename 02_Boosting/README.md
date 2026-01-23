# 🚀 Gradient Boosting Classifier: From Scratch

## 📌 Module Overview
This module breaks down the **Gradient Boosting Machine (GBM)** algorithm for classification. Unlike standard tutorials that simply import `sklearn`, this project implements the algorithm **from scratch** to demonstrate the underlying mathematics and optimization logic.

It is based on **Jerome Friedman's** classic paper: *"Greedy Function Approximation: A Gradient Boosting Machine"*.

## 📂 Files in this Module

| File Name | Description |
|:--- |:--- |
| **`gradient-boosting-classification.ipynb`** | **The Theory.** A deep dive into the mathematical derivation of Gradient Boosting. It explains how we treat model training as an optimization problem in function space using pseudo-residuals. |
| **`gradient-boosting-classification-demo.ipynb`** | **The Code.** A step-by-step Python implementation of the algorithm on a custom dataset without using the `GradientBoostingClassifier` library. |

---

## 🧠 Part 1: The Math (Theory)
**Focus:** gradient-boosting-classification.ipynb`

This notebook explains the core intuition behind Boosting:
1.  **Additive Modeling:** We build the model sequentially, adding weak learners (Decision Trees) one by one.
2.  **Pseudo-Residuals:** Instead of correcting the absolute error, we train new trees to predict the *negative gradient* of the loss function.
3.  **Log-Odds Transformation:** For classification, we work with log-odds to map probability predictions to a continuous space suitable for regression trees.

> **Key Formula:**
> $F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$
> *Where $F_m$ is the new model, $\nu$ is the learning rate, and $h_m$ is the new tree trained on residuals.*

---

## 💻 Part 2: The Implementation (Code)
**Focus:** `gradient-boosting-classification-demo.ipynb`

We implement the algorithm on a **Student Placement Dataset** (`cgpa`, `iq` $\rightarrow$ `is_placed`).

### Algorithm Steps Implemented:
1.  **Initialization:** Calculate the initial log-odds prediction for the entire dataset.
2.  **Loop (Boosting Rounds):**
    * **Calculate Residuals:** Compute the difference between the actual value (0 or 1) and the predicted probability.
    * **Train Tree:** Fit a Decision Tree Regressor to these residuals.
    * **Calculate Leaf Output:** Apply the transformation formula to convert leaf values back into log-odds adjustments.
    * **Update Model:** Add the new tree's predictions (scaled by learning rate) to the previous predictions.
3.  **Prediction:** Convert final log-odds to probability using the Sigmoid function.

## 🛠️ Dependencies
* `numpy`
* `pandas`
* `scikit-learn` (Only used for the base DecisionTreeRegressor and splitting data)
* `matplotlib` (For visualization)

## 🚀 How to Run
1.  Navigate to this folder:
    ```bash
    cd Machine-Learning-Concepts/02_Boosting
    ```
2.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `gradient-boosting-classification-demo.ipynb` to see the code in action.


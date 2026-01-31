# CatBoost

## 1. The Concept
CatBoost is a high-performance open-source library for gradient boosting on decision trees. Its name comes from **"Category"** + **"Boosting"**.

### The "Categorical" Superpower
In most algos (XGBoost/Sklearn), you must convert text categories ("Red", "Blue") into numbers manually. CatBoost does this **automatically** during training using smart statistical encodings (Target Encoding).

## 2. Key Advantages
* **No Preprocessing:** Handles categorical features automatically.
* **Symmetric Trees:** Builds balanced trees, which makes prediction extremely fast.
* **Robustness:** Works well with default parameters (less tuning needed).

## 3. Resources
* [CatBoost Documentation](https://catboost.ai/)
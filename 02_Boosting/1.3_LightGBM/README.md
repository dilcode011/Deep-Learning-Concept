# LightGBM

## 1. The Concept
LightGBM is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be distributed and efficient.

### The "Leaf-Wise" Growth Strategy
Most algorithms (like XGBoost) grow trees **Level-Wise** (row by row). LightGBM grows trees **Leaf-Wise** (it finds the single leaf with the highest error and splits it).
* **Pros:** Converges much faster.
* **Cons:** Can overfit on small datasets (requires `max_depth` parameter).



## 2. Key Advantages
* **Speed:** Faster training speed and higher efficiency.
* **Memory:** Lower memory usage (uses histogram-based algorithms).
* **Large Datasets:** Capable of handling large-scale data.

## 3. Resources
* [LightGBM Documentation](https://lightgbm.readthedocs.io/)
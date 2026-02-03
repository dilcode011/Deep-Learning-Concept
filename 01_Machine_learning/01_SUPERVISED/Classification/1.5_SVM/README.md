# Support Vector Machine (SVM)

## 1. The Concept
Support Vector Machine (SVM) is a powerful supervised learning algorithm used for **Classification** (and sometimes Regression).

The goal of SVM is to find the **best boundary** (called a Hyperplane) that separates the data classes with the **widest possible margin**.

### Key Terms
* **Hyperplane:** The line (in 2D) or plane (in 3D) that separates the classes.
* **Support Vectors:** The specific data points closer to the hyperplane that "support" or define the orientation of the boundary.
* **Margin:** The distance between the hyperplane and the nearest data point from either class. We want to *maximize* this margin.


## 2. The "Kernel Trick" (SVM's Superpower)
What if the data is not linearly separable (e.g., a red circle inside a blue ring)? You can't draw a straight line to separate them.

SVM solves this by projecting the data into a **Higher Dimension** (3D or more) where a separation *is* possible. This math magic is called the **Kernel Trick**.

Common Kernels:
1.  **Linear:** Standard straight line separation.
2.  **Polynomial:** Curved lines.
3.  **RBF (Radial Basis Function):** Creates complex, circular, or island-like boundaries (most popular).

## 3. The Math (Simplified)
**Objective Function:**
We want to minimize $|w|$ (to maximize margin) subject to the constraint that all points are classified correctly:
$$y_i (w \cdot x_i + b) \ge 1$$

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: SVM](https://scikit-learn.org/stable/modules/svm.html)

### 📺 Videos
* [StatQuest: SVM Main Ideas](https://www.youtube.com/watch?v=efR1C6CvhmE)
* [MIT OpenCourseWare: SVM Explained](https://www.youtube.com/watch?v=_PwhiWxHK8o)
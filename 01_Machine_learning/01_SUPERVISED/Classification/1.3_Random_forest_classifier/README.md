# Random Forest Classifier

## 1. The Concept
Random Forest is an **Ensemble Learning** method that operates by constructing a multitude of Decision Trees at training time.

### The "Wisdom of the Crowd" Analogy
Imagine you want to buy a new phone:
* **Decision Tree:** You ask one tech expert. If they are biased or make a mistake, you get bad advice.
* **Random Forest:** You ask 100 random people. Even if some are wrong, the majority vote is likely to be the correct choice.



## 2. How It Works (Bagging)
Random Forest uses a technique called **Bagging** (Bootstrap Aggregating):

1.  **Bootstrapping:** We create multiple random subsets of the training data (sampling *with replacement*).
2.  **Building Trees:** We build a Decision Tree for each subset.
    * *Crucial Difference:* At each split in the tree, the model only considers a **random subset of features**, not all of them. This ensures the trees are different from each other (uncorrelated).
3.  **Aggregating:**
    * **Classification:** Majority Vote (e.g., 80 trees say "Yes", 20 say "No" -> Result: "Yes").
    * **Regression:** Average of all predictions.



## 3. Random Forest vs. Decision Tree
| Feature | Single Decision Tree | Random Forest |
| :--- | :--- | :--- |
| **Overfitting** | High risk (memorizes data) | Very Low risk (averages out errors) |
| **Interpretability** | Easy to visualize | Hard (Black Box) |
| **Speed** | Fast to train | Slower (builds many trees) |
| **Performance** | Good | **Excellent** |

## 4. Resources
### 📖 Documentation
* [Scikit-Learn: Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 📺 Videos
* [StatQuest: Random Forests Part 1](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
* [Computerphile: Random Forest Implementation](https://www.youtube.com/watch?v=loNcrMjYh64)
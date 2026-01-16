# Introduction to Machine Learning

## 1. What is Machine Learning?
Machine Learning (ML) is a subset of AI where computers learn from data without being explicitly programmed for specific rules.

* **Traditional Programming:** You give the computer **Rules** + **Data**, and it gives you **Answers**.
* **Machine Learning:** You give the computer **Answers (Data)**, and it finds the **Rules**.



## 2. Types of Machine Learning
Machine Learning is generally categorized into three main types:

### A. Supervised Learning (The "Teacher" Approach)
The model learns from labeled data. You tell the model: *"Here is an image of a cat, and here is an image of a dog. Now learn the difference."*
* **Regression:** Predicting a continuous number (e.g., House Prices, Stock Value).
- [x] 1.1 Linear Regression
- [x] 1.2 Ridge Regression
- [x] 1.3 Lasso Regression
- [x] 1.4 Elastic Net
* **Classification:** Predicting a category (e.g., Spam or Not Spam, Cat or Dog).
- [x] 1.1 Logistic Regression
- [x] 1.2 Decision Tree Classifier
- [x] 1.3 Support Vector Machine (SVM)
- [x].1.4 Random Forest Classifier
- [x] 1.5 K-Nearest Neighbors (KNN)
- [x] 1.6 Naive Bayes

### B. Unsupervised Learning (The "Self-Study" Approach)
The model learns from unlabeled data. You give the model raw data and ask it to find patterns.
* **Clustering:** Grouping similar items (e.g., Customer Segmentation).
* **Dimensionality Reduction:** Compressing data while keeping the important structure.

### C. Reinforcement Learning (The "Trial and Error" Approach)
An agent learns to make decisions by performing actions and receiving rewards or penalties (e.g., Teaching a robot to walk, Chess engines).



## 3. Key Terminology
* **Dataset:** The collection of data used for learning.
* **Features (X):** The input variables (e.g., square footage, number of bedrooms).
* **Labels (y):** The output variable we want to predict (e.g., price of the house).
* **Training Set:** The data the model studies to learn.
* **Test Set:** The data hidden from the model, used to evaluate how well it learned.
* **Model:** The mathematical engine (algorithm) that learns the patterns.

## 4. The 7 Steps of Machine Learning
1.  **Data Gathering:** Collecting the raw material.
2.  **Data Preprocessing:** Cleaning and formatting the data.
3.  **Feature Engineering:** Selecting the most relevant variables.
4.  **Model Selection:** Choosing the right algorithm (e.g., Linear Regression, Random Forest).
5.  **Training:** The model learns from the data.
6.  **Evaluation:** Testing accuracy on new data.
7.  **Deployment:** Putting the model into production.
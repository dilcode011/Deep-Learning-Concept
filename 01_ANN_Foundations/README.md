# Part 1: ANN Foundations

Welcome to the Artificial Neural Network (ANN) module. This section is designed to take you from the basic mathematical concept of a single neuron to building full-scale deep learning models using industry-standard frameworks.

##  Folder Structure & Learning Path

To learn effectively, follow the files in this order:

### 1. [01_The_Neuron](./01_The_Neuron/simple_neuron.py)
**Concept:** The Mathematical Building Block.
* Understands how **Weights ($w$)** and **Biases ($b$)** interact with input data.
* Demonstrates the linear transformation: $Z = W \cdot X + b$.
* **Key Takeaway:** A neuron is just a mathematical function that scales and shifts data.

### 2. [02_Activation_Functions](./02_Activation_Functions/activations_visualizer.py)
**Concept:** Introducing Non-Linearity.
* Visualizes **ReLU**, **Sigmoid**, and **Tanh**.
* **Why it matters:** Without activation functions, an ANN is just a linear regression model. These functions allow the network to learn complex patterns (curves, shapes, and decision boundaries).

### 3. [03_Data_Preprocessing](./03_Data_Preprocessing/scaler_pipeline.py)
**Concept:** Preparing Data for Deep Learning.
* Covers **Label Encoding** and **Standard Scaling**.
* **Why it matters:** ANNs are sensitive to the scale of data. If "Salary" is 50,000 and "Age" is 20, the network will focus only on Salary. Scaling ensures both features contribute equally to learning.

### 4. [04_Optimization_Training](./04_Optimization_Training/ann_with_keras.py)
**Concept:** The Framework Implementation.
* Uses **TensorFlow/Keras** to build a multi-layer architecture.
* Introduces the **Adam Optimizer** and **Binary Cross-Entropy** loss.
* **Key Takeaway:** This is where theory meets practice—building a model that can actually predict outcomes.



# 🔀 Keras Functional API Explained

## 📌 Overview
This module demonstrates the **Keras Functional API**, a way to build Deep Learning models that is more flexible than the standard `Sequential` API.

While `Sequential` models are limited to a linear stack of layers, the **Functional API** treats every layer as a function that takes a tensor and returns a tensor. This allows for complex architectures like **ResNet** (skip connections) or Multi-Input models.

## 🆚 Sequential vs. Functional

### 1. The Sequential Approach (Linear)
*Best for simple stacks of layers.*
```python
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
#  CNN Mechanics: Padding, Strides, and Pooling

##  Overview
This notebook demonstrates the fundamental mechanics of **Convolutional Neural Networks (CNNs)**. It specifically isolates and visualizes how different hyperparameters—**Padding**, **Strides**, and **Pooling**—affect the output dimensions (shape) of feature maps.

Understanding these concepts is crucial for designing custom deep learning architectures and debugging "shape mismatch" errors in TensorFlow/Keras.

##  Concepts Demonstrated

### 1. Padding (`padding='valid'` vs `padding='same'`)
The notebook compares two types of padding in a `Conv2D` layer:
* **Valid Padding:** No padding is added. The filter only operates on valid pixels, causing the output feature map to shrink.
    * *Observation:* Input `(28, 28)` $\rightarrow$ Output `(26, 26)` (with 3x3 kernel).
* **Same Padding:** Zero-padding is added to the borders to ensure the output size matches the input size.
    * *Observation:* Input `(28, 28)` $\rightarrow$ Output `(28, 28)`.

### 2. Strides
Demonstrates the effect of increasing the step size of the filter.
* **Configuration:** `strides=(2,2)`
* **Effect:** The filter skips pixels, effectively acting as a downsampler and reducing the computational cost.
* *Observation:* Input `(28, 28)` $\rightarrow$ Output `(14, 14)`.

### 3. Pooling Layers
Demonstrates `MaxPooling2D` to reduce dimensionality while retaining the most important features.

##  Tech Stack
* **Python**
* **TensorFlow / Keras** (Sequential API)
* **NumPy**

##  Files
* `keras-padding-strides.ipynb` (or `CNN_Padding_Strides_Demo.ipynb`): The main Jupyter Notebook containing the experiments and `model.summary()` outputs.

2.  **Run the Notebook:**
    Open the file in **Google Colab** or **Jupyter Notebook**.
3.  **Analyze Results:**
    Run the cells and look at the `Output Shape` column in the `model.summary()` tables to see the mathematical transformations in action.

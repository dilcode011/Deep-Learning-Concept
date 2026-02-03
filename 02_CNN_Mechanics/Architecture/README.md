---

##  Project 2: LeNet-5 Architecture Implementation
**File:** `02_Classic_Architectures/LeNet_Implementation.ipynb`

###  Historical Context
LeNet-5, proposed by Yann LeCun in 1998, is one of the earliest Convolutional Neural Networks. It paved the way for modern Computer Vision by successfully classifying handwritten digits on bank cheques (MNIST).

###  Implementation Details
This notebook recreates the classic architecture using modern TensorFlow/Keras.

#### Key Architectural Features Observed:
* **Average Pooling:** Unlike modern CNNs that typically use *Max Pooling*, LeNet-5 uses *Average Pooling* to downsample feature maps.
* **Tanh Activation:** The original paper used `tanh` (or sigmoid) activations rather than the now-standard `ReLU`.
* **Structure:** The specific sequence of `Conv2D` $\rightarrow$ `AveragePooling2D` $\rightarrow$ `Conv2D` $\rightarrow$ `AveragePooling2D` $\rightarrow$ `Flatten` $\rightarrow$ `Dense`.

#### Results
* **Dataset:** Trained on the **MNIST** dataset (60,000 training images).
* **Parameter Count:** The model is extremely lightweight (~60,000 parameters) compared to modern networks like VGG or ResNet (millions of parameters).

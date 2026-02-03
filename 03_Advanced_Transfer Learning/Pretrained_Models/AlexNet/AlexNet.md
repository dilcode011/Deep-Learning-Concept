#  AlexNet: The Deep Learning Revolution

**Paper:** [ImageNet Classification with Deep Convolutional Neural Networks (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  
**Creators:** Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (University of Toronto)

---

##  Introduction
**AlexNet** is the model that started the modern Deep Learning boom.

In 2012, it competed in the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** and achieved a top-5 error rate of **15.3%**, more than **10.8 percentage points lower** than the runner-up. This massive gap proved to the world that **Deep Convolutional Neural Networks (CNNs)** trained on **GPUs** were the future of computer vision.

---

##  Architecture Overview
AlexNet is much deeper than its predecessor, LeNet-5. It consists of **8 layers**:
* **5 Convolutional Layers** (for feature extraction)
* **3 Fully Connected Layers** (for classification)

### The "Sandwich" Structure
1.  **Input:** 227 x 227 x 3 image.
2.  **Conv1:** 96 filters of size 11x11, stride 4.
3.  **Conv2:** 256 filters of size 5x5.
4.  **Conv3:** 384 filters of size 3x3.
5.  **Conv4:** 384 filters of size 3x3.
6.  **Conv5:** 256 filters of size 3x3.
7.  **FC6:** 4096 neurons.
8.  **FC7:** 4096 neurons.
9.  **Output (FC8):** 1000 neurons (Softmax).

---

##  Key Innovations
AlexNet introduced several techniques that are now standard in almost every neural network:

### 1. ReLU Activation (Rectified Linear Unit)
Before AlexNet, models used `Tanh` or `Sigmoid` activation functions, which were slow to train. AlexNet used **ReLU**:
$$f(x) = \max(0, x)$$
* **Benefit:** It accelerated training by **6x** compared to Tanh.

### 2. Dropout
To prevent overfitting (memorizing the training data), AlexNet introduced **Dropout**.
* **How it works:** During training, it randomly "turns off" 50% of the neurons in the fully connected layers. This forces the network to learn more robust features.

### 3. GPU Training
The model was too large to train on a CPU. AlexNet was specifically implemented to run on two **NVIDIA GTX 580 GPUs** in parallel.

### 4. Data Augmentation
They artificially expanded the dataset by:
* Randomly cropping images.
* Flipping images horizontally.
* Changing pixel intensities (color jittering).

---

### ⚔️ Comparison: LeNet-5 vs. AlexNet

| Feature | LeNet-5 (1998) | AlexNet (2012) |
| :--- | :--- | :--- |
| **Layers** | 5 | 8 |
| **Activation** | Tanh / Sigmoid | ReLU |
| **Pooling** | Average Pooling | Max Pooling |
| **Device** | CPU | GPU |
| **Image Size** | 32x32 (Grayscale) | 227x227 (Color) |
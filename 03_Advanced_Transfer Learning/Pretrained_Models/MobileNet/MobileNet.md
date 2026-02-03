# 📱 MobileNet: AI for Mobile Devices

**Paper:** [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision (2017)](https://arxiv.org/abs/1704.04861)  
**Focus:** Speed and Efficiency (Low Latency)

---

## 📖 Introduction
Most models (ResNet, VGG) are too heavy to run on a phone. **MobileNet** was designed specifically for mobile and embedded vision applications. It is small, fast, and low-power.

## 💡 Key Innovation: Depthwise Separable Convolutions
MobileNet replaces standard convolutions with a smarter trick that breaks the operation into two steps:
1.  **Depthwise Conv:** Filters apply to a single channel only (Red, Green, Blue separately).
2.  **Pointwise Conv:** A 1x1 convolution combines the outputs.

### 🧠 The Math: The Speedup
Standard Convolution cost: $D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$
MobileNet Convolution cost: $D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$

* **Result:** This reduces computation by **8 to 9 times** with only a tiny drop in accuracy.

## 💻 Code Snippet (Keras)
```python
from tensorflow.keras.applications import MobileNet

# Load MobileNet - Small and Fast!
model = MobileNet(weights='imagenet')

# MobileNet is very lightweight (~16MB) compared to VGG (~500MB)
model.summary()
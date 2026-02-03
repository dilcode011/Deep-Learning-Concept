# 🌈 Inception (GoogLeNet): Going Deeper with Width
 
**Paper:** [Going Deeper with Convolutions (2014)](https://arxiv.org/abs/1409.4842)  
**Winner:** ILSVRC 2014 (Beat VGG16)

---

## 📖 Introduction
While VGG focused on making networks *deeper*, Google asked: "Can we make them *wider*?"
**Inception (v1)**, also known as GoogLeNet, introduced a complex architecture that allows the network to choose the best filter size automatically.

## 💡 Key Innovation: The Inception Module
In a normal CNN, you have to decide: "Should I use a 3x3 filter? Or a 5x5? Or Max Pooling?"
The **Inception Module** says: **"Do them all."**

It performs 1x1, 3x3, and 5x5 convolutions *in parallel* and concatenates the results. This lets the model capture both fine details (small filters) and big patterns (large filters) at the same layer.

### 🧠 The Math: 1x1 Convolutions
To prevent the computation from exploding, they used **1x1 convolutions** to reduce dimensions (like compression) before the expensive 3x3 and 5x5 layers.
* This reduced parameters from ~138M (VGG) to just **~4M** (GoogLeNet).


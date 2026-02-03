#  ResNet (Residual Networks): Breaking the Depth Barrier

**Author:** Dilpreet Singh  
**Paper:** [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)  
**Creators:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

---

##  Introduction
**ResNet** (Residual Network) is arguably the most groundbreaking architecture in modern computer vision. In 2015, it won the ILSVRC (ImageNet) competition with an incredible error rate of **3.57%** (beating human performance).

Before ResNet, training deep neural networks was impossible due to the **Vanishing Gradient Problem**. ResNet allowed engineers to train networks with **152+ layers** (compared to VGG's 19 layers) without losing performance.

---

##  The Problem: "The Degradation"
Intuitively, we think "Deeper is Better." If a 20-layer network works well, a 50-layer network should work better, right?

**Wrong.**
Researchers found that as you add more layers, the accuracy eventually **saturates and then degrades rapidly**. This wasn't due to overfitting; it was an optimization issue. The signal (gradient) carrying the "correction" from the output back to the input would become so small that it vanished before reaching the early layers. The early layers essentially stopped learning.

---

##  The Solution: Residual Learning (Skip Connections)
ResNet introduced the concept of **Skip Connections** (or Identity Shortcuts).

Instead of forcing a layer to learn the entire transformation of the data, ResNet forces the layer to learn only the "Residual" (the difference or the edit) needed to improve the data.

###  The Mathematical Intuition
Let $H(x)$ be the "optimal" mapping we want the network to learn.
* **Traditional CNNs** try to learn $H(x)$ directly.
* **ResNet** breaks it down. It lets the layers learn a residual function $F(x) := H(x) - x$.
* Therefore, the original mapping becomes:
    $$H(x) = F(x) + x$$

**Why does this work?**
If the network is already perfect and needs no more changes, it's much easier for the weights to shrink to zero ($F(x) \to 0$) so that the output is just the identity $x$ ($0 + x = x$). In a traditional network, learning the identity function from scratch is very difficult.

---

##  Architecture: The Residual Block
A ResNet is built by stacking these "Residual Blocks."

**Structure of one block:**
1.  Input $x$
2.  **Weight Layer** (Conv2D + Batch Norm + ReLU)
3.  **Weight Layer** (Conv2D + Batch Norm)
4.  **Addition:** $F(x) + x$ (The Skip Connection adds the original input to the result)
5.  **Relu Activation**


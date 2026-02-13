#  Bahdanau vs. Luong Attention

Before the Transformer architecture existed, Attention was designed to fix the information bottleneck in Seq2Seq models (like LSTMs used for translation). The two foundational mechanisms that made this possible were created by **Dzmitry Bahdanau (2014)** and **Minh-Thang Luong (2015)**.

While both mechanisms allow a Decoder to "look back" at the Encoder, they differ significantly in **how they calculate the attention score**.

---

## 1. Bahdanau Attention (Additive Attention)
Introduced in 2014, this was the very first attention mechanism. 

### How it works:
Bahdanau uses a **small Feed-Forward Neural Network (Dense Layer)** to calculate the similarity score between the Decoder's state and the Encoder's states. Because it adds the states together inside the network, it is called **Additive Attention**.

### The Math:
$$Score(s_{t-1}, h_i) = V^T \tanh(W_1 s_{t-1} + W_2 h_i)$$

* $s_{t-1}$: The **previous** hidden state of the Decoder.
* $h_i$: The hidden states of the Encoder.
* $W_1, W_2, V$: Trainable weight matrices.

*Note: Because it uses a neural network layer to calculate every single score, it is computationally expensive.*

---

## 2. Luong Attention (Multiplicative Attention)
Introduced in 2015, Luong sought to speed up Bahdanau's approach. Instead of a neural network layer, Luong uses **matrix multiplication** (dot products) to calculate the scores. Because of this, it is called **Multiplicative Attention**.

### How it works:
Luong realized that if the Decoder state and the Encoder state are vectors, you can find their similarity simply by multiplying them together. 

### The Math (General Variation):
$$Score(s_t, h_i) = s_t^T \cdot W \cdot h_i$$

* $s_t$: The **current** hidden state of the Decoder.
* $h_i$: The hidden states of the Encoder.
* $W$: A trainable weight matrix to align dimensions.

---

##  Summary of Differences

| Feature | Bahdanau Attention | Luong Attention |
| :--- | :--- | :--- |
| **Also known as** | Additive Attention | Multiplicative / Dot-Product Attention |
| **Scoring Method** | Feed-Forward Neural Network | Dot Product / Matrix Multiplication |
| **Speed** | Slower (calculates through Dense layers) | **Faster** (optimized matrix multiplication) |
| **Decoder State Used** | Uses the **previous** hidden state ($s_{t-1}$) | Uses the **current** hidden state ($s_t$) |
| **Code Implementation**| Requires `tf.keras.layers.Dense` | Uses `tf.matmul` |

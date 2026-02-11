#  The Gated Recurrent Unit (GRU)

Introduced in 2014 by Cho et al., the **Gated Recurrent Unit (GRU)** is a simplified version of the LSTM. It solves the vanishing gradient problem just like an LSTM but with a more streamlined architecture.

Think of the GRU as the "efficient younger sibling" of the LSTM. It often achieves similar performance but is computationally cheaper and faster to train.

## 1. Key Differences from LSTM

The GRU simplifies the LSTM by combining concepts:

1.  **No Cell State:** It merges the Cell State ($C_t$) and Hidden State ($h_t$) into a single Hidden State ($h_t$).
2.  **Two Gates instead of Three:**
    * **Update Gate ($z_t$):** Combines the Forget and Input gates into one. Decides how much of the past to keep and how much new info to add.
    * **Reset Gate ($r_t$):** Decides how much of the past information to forget before calculating the new candidate state.

## 2. The Architecture

While an LSTM has a complex flow with 3 sigmoids and 2 tanh functions, a GRU is leaner.

### The Equations

| Component | Equation | Function |
| :--- | :--- | :--- |
| **Update Gate** | $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ | Decides: "How much of the past should I keep?" (Close to 1 = keep past, Close to 0 = update with new). |
| **Reset Gate** | $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$ | Decides: "How much of the past should I ignore to compute the *new* candidate?" |
| **Candidate State** | $\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$ | Creates the new potential hidden state. Notice how $r_t$ allows the network to drop past info *before* the tanh. |
| **Final Hidden State** | $h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$ | The linear interpolation. It takes a percentage of the old state and a percentage of the new candidate. |

## 3. GRU vs. LSTM: Which one to use?

There is no clear winner, but here are the general rules of thumb:

| Feature | **GRU** | **LSTM** |
| :--- | :--- | :--- |
| **Complexity** | Simpler (fewer parameters). | Complex (more parameters). |
| **Training Speed** | **Faster.** Great for large datasets or limited compute. | Slower per epoch. |
| **Data Size** | Often better on **smaller datasets** (less prone to overfitting). | Often better on **large, complex datasets** (more capacity to learn). |
| **Long Sequences**| Good, but theoretically slightly weaker at very long dependencies. | **Stronger** at remembering very long-term dependencies due to the separate Cell State. |

## 4. Keras Implementation

Switching between them in code is as simple as changing the layer name.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Embedding, Dense

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128))

# Using GRU instead of LSTM
# It accepts the same arguments (units, return_sequences, etc.)
model.add(GRU(64, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
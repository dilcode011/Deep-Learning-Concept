# The Mathematics Behind LSTM Gates

This document breaks down the precise mathematical operations that occur inside an LSTM unit during a single forward pass at time step $t$.

## Notation Guide
* $x_t$: Input vector at time $t$.
* $h_{t-1}$: Hidden state from the previous time step.
* $C_{t-1}$: Cell state from the previous time step.
* $\sigma$: Sigmoid function (maps to $[0, 1]$).
* $\tanh$: Hyperbolic tangent function (maps to $[-1, 1]$).
* $W$: Weight matrices (parameters learned during training).
* $b$: Bias vectors.

---

## Step 1: The Forget Gate ($f_t$)
**Goal:** Decide what information from the previous cell state $C_{t-1}$ should be discarded or kept.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

* We concatenate the previous hidden state and current input.
* If $f_t \approx 0$, the information is forgotten.
* If $f_t \approx 1$, the information is kept.

---

## Step 2: The Input Gate ($i_t$ and $\tilde{C}_t$)
**Goal:** Decide what new information to store in the cell state. This happens in two parts:

1.  **Identify which values to update ($i_t$):**
    $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

2.  **Create a vector of new candidate values ($\tilde{C}_t$):**
    $$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

---

## Step 3: Update the Cell State ($C_t$)
**Goal:** Actually update the long-term memory.

$$C_t = (f_t * C_{t-1}) + (i_t * \tilde{C}_t)$$

* **Part 1 ($f_t * C_{t-1}$):** Forget the old things we decided to forget.
* **Part 2 ($i_t * \tilde{C}_t$):** Add the new candidate values, scaled by how much we decided to update each state.
* *Note:* The $*$ operator here represents element-wise multiplication (Hadamard product).

---

## Step 4: The Output Gate ($o_t$ and $h_t$)
**Goal:** Decide what the next Hidden State (prediction) should be.

1.  **Decide what parts of the Cell State to output ($o_t$):**
    $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

2.  **Calculate the new Hidden State ($h_t$):**
    $$h_t = o_t * \tanh(C_t)$$

* We push the Cell State through $\tanh$ (to force values between -1 and 1).
* We multiply it by the output gate $o_t$ to only output the parts we decided were relevant.

---

## Summary of Equations

| Step | Equation |
| :--- | :--- |
| **Forget** | $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ |
| **Input (Gate)** | $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ |
| **Input (Candidate)** | $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$ |
| **Cell Update** | $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$ |
| **Output (Gate)** | $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$ |
| **Hidden State** | $h_t = o_t * \tanh(C_t)$ |
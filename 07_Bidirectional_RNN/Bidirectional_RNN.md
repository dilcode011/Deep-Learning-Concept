# ↔ Bidirectional RNNs (BiRNN)

> **"Why look only at the past, when the future holds the answer?"**

Standard Recurrent Neural Networks (RNNs) and LSTMs have a limitation: they process data linearly from **left to right** (time $t=0 \to t=T$). This means at any given word, the network only knows what came *before* it.

**Bidirectional RNNs (BiRNNs)** solve this by processing the data **twice** simultaneously: once from start to end, and once from end to start. This gives the network context from both the **past** and the **future**.

##  The Concept: Why do we need it?

Consider this fill-in-the-blank problem:

> **"He said ______ to the crowd."**

1.  **Standard RNN (Forward only):**
    * Reads: *"He said..."*
    * Prediction: *"Hello"? "Nothing"? "Yes"?*
    * *Problem:* Without seeing the end of the sentence ("to the crowd"), it's guessing blindly.

2.  **Bidirectional RNN:**
    * **Forward Layer** reads: *"He said..."*
    * **Backward Layer** reads: *"...crowd the to"*
    * **Combined View:** It knows someone is speaking **to a crowd**.
    * Prediction: *"Hello"* or *"Speech"* (High confidence).

##  The Architecture

A BiRNN is essentially **two independent RNNs** stacked on top of each other:

1.  **Forward RNN ($\overrightarrow{RNN}$):** Reads input sequence from $x_1$ to $x_T$.
2.  **Backward RNN ($\overleftarrow{RNN}$):** Reads input sequence from $x_T$ to $x_1$.

At every time step $t$, the final output $y_t$ is a combination (usually concatenation) of the hidden states from both RNNs.

```mermaid
graph TD
    subgraph Time Step t
    Input((x_t)) --> Forward[Forward Layer h->]
    Input --> Backward[Backward Layer <-h]
    Forward --> Concat[Concatenation]
    Backward --> Concat
    Concat --> Output((y_t))
    end
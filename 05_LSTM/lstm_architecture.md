#  The Architecture of LSTMs

Standard Recurrent Neural Networks (RNNs) have a major flaw: they have a "short-term memory." If a sequence is long enough, they forget the beginning by the time they reach the end. This is due to the **Vanishing Gradient Problem**.

**Long Short-Term Memory (LSTM)** networks were designed specifically to solve this. They introduce an internal mechanism called the **Cell State** (a "memory highway") that allows information to flow through the entire sequence almost unchanged.

## 1. The Core Concept: Cell State vs. Hidden State

The LSTM unit (or "cell") maintains two separate states at each time step $t$:

1.  **Cell State ($C_t$):** The "Long-Term Memory."
    * Think of this as a conveyor belt running straight down the entire chain.
    * It carries the core information (e.g., the subject of the sentence) with very minor linear interactions.
    * It is very easy for information to flow along it unchanged.

2.  **Hidden State ($h_t$):** The "Short-Term Memory" (and Output).
    * This is the output used for prediction at the current step.
    * It is a filtered version of the Cell State, containing only what is relevant for the *immediate* next step.

## 2. The Three Gates

The LSTM regulates the flow of information using structures called **Gates**. A gate is simply a sigmoid neural net layer ($0$ to $1$) and a multiplication operation.

| Gate | Name | Function | Analogy |
| :--- | :--- | :--- | :--- |
| **1** | **Forget Gate** | Decides what to throw away from the Cell State. | "The subject changed from 'He' to 'She', so forget the old gender." |
| **2** | **Input Gate** | Decides what *new* information to store in the Cell State. | "Add the new gender info to the memory." |
| **3** | **Output Gate** | Decides what to output (Hidden State) based on the updated Cell State. | "Since the subject is 'She', the next verb should be singular form." |

## 3. Visual Diagram

Here is the flow of data through a single LSTM cell at time step $t$.

```mermaid
graph TD
    subgraph LSTM Cell
    Input[Input x_t] --> Sigmoid1[Forget Gate (Sigmoid)]
    Prev_H[Prev Hidden h_{t-1}] --> Sigmoid1

    Input --> Sigmoid2[Input Gate (Sigmoid)]
    Prev_H --> Sigmoid2
    Input --> Tanh1[Candidate Memory (Tanh)]
    Prev_H --> Tanh1

    Prev_C[Prev Cell State C_{t-1}] --> Mult1((x))
    Sigmoid1 --> Mult1

    Mult1 --> Plus((+))
    Sigmoid2 --> Mult2((x))
    Tanh1 --> Mult2
    Mult2 --> Plus

    Plus --> New_C[New Cell State C_t]
    
    Plus --> Tanh2[Tanh]
    Input --> Sigmoid3[Output Gate (Sigmoid)]
    Prev_H --> Sigmoid3
    
    Tanh2 --> Mult3((x))
    Sigmoid3 --> Mult3
    Mult3 --> New_H[New Hidden State h_t]
    end
# 🔄 Recurrent Neural Networks (RNN)

##  Overview
This module explores **Recurrent Neural Networks (RNNs)**, a class of Deep Learning models designed to handle sequential data.

Unlike standard Feed-Forward Networks (where inputs are independent), RNNs have "memory." They process inputs step-by-step, maintaining an internal state (hidden state) that captures information from previous time steps.

##  The Core Concept: "Memory"
Imagine reading a sentence: *"The food was bad, but the service was good."*
* To understand the word *"good"* at the end, you need to remember that it refers to *"service"*, not *"food"*.
* **Standard Networks:** See each word in isolation. They might see "bad" and "good" and get confused.
* **RNNs:** Process the sentence left-to-right, passing context forward.

##  The Challenge: Vanishing Gradients
Standard RNNs suffer from short-term memory. As the sequence gets longer (e.g., a paragraph), the gradients (signals used to update weights) become smaller and smaller until they vanish. This makes it hard for the network to remember things from the beginning of the sequence.

* **Solution:** We typically use advanced RNN variants like **LSTM (Long Short-Term Memory)** or **GRU (Gated Recurrent Unit)** to handle long-term dependencies.


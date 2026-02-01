## 🔄 Data Augmentation Strategy

To improve the model's generalization and address the overfitting observed in early trials (where Training Accuracy reached ~95% while Validation Accuracy stalled at ~83%), this project implements **Data Augmentation**.

By artificially expanding the training dataset, the model learns to recognize features (like ears, tails, and fur patterns) regardless of orientation or size, rather than memorizing specific pixels.

### Techniques Applied
The following transformations are applied to the training images dynamically using `keras.layers`:
* **Random Flip (Horizontal):** Simulates the subject facing the opposite direction.
* **Random Rotation (0.1):** Rotates the image slightly (up to ~10%) to account for camera angles.
* **Random Zoom (0.1):** Zooms in/out slightly to handle variations in subject distance.

### Implementation Code
```python
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])
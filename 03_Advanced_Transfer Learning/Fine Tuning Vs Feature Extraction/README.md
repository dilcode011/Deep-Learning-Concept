#  Transfer Learning: Dogs vs. Cats Classification

##  Project Overview
This repository demonstrates a comprehensive study of **Transfer Learning** techniques using TensorFlow/Keras. The goal is to classify images of dogs and cats using a pre-trained **VGG16** model.

The project is structured as a progressive experiment, starting from a basic baseline and evolving into advanced fine-tuning to maximize accuracy and reduce overfitting.

##  Experiments & Progression

### 1. Feature Extraction (No Augmentation)
* **File:** `01_Feature_Extraction/No_Augmentation.ipynb`
* **Method:** The convolutional base of VGG16 is loaded and frozen. A custom Dense classifier is added on top.
* **Observation:** The model learns quickly but suffers from **overfitting**. Training accuracy reaches ~100%, while validation accuracy stalls, indicating the model is memorizing the training data.

### 2. Feature Extraction + Data Augmentation
* **File:** `01_Feature_Extraction/With_Augmentation.ipynb`
* **Method:** To combat overfitting, real-time data augmentation (rotation, zoom, flips) is introduced using `ImageDataGenerator`.
* **Observation:** The model generalizes much better. The gap between training and validation accuracy narrows significantly, providing a robust baseline.

### 3. Fine-Tuning (The "Power Move")
* **File:** `02_Fine_Tuning/Fine_Tuning_VGG16.ipynb`
* **Method:** 1.  Start with the converged model from the previous step.
    2.  **Unfreeze** the top few layers of the VGG16 base.
    3.  Re-train the model with a **very low learning rate** (e.g., `1e-5`).
* **Observation:** This allows the specialized high-level features of the pre-trained network to adapt specifically to the "Dogs vs. Cats" dataset, pushing validation accuracy to its highest point (typically >90%).

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Pre-trained Model:** VGG16 (ImageNet weights)
* **Data Handling:** Keras `ImageDataGenerator` / `image_dataset_from_directory`
* **Dataset:** Kaggle Dogs vs. Cats

##  How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Transfer-Learning-Experiments.git](https://github.com/YOUR_USERNAME/Transfer-Learning-Experiments.git)
    ```
2.  **Download the Dataset:**
    The notebooks use the Kaggle API to download the `dogs-vs-cats` dataset. Ensure you have your `kaggle.json` key ready or download the data manually.
3.  **Run the Notebooks:**
    Open the notebooks in Google Colab (recommended for GPU support) to replicate the training process.
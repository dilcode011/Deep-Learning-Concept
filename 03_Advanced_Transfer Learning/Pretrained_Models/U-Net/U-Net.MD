# 🩺 U-Net: Biomedical Image Segmentation
 
**Paper:** [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597)  
**Domain:** Medical Imaging (Tumor detection, Cell segmentation)

---

## 📖 Introduction
Classification models (like ResNet) say *"This is a cat."*
Detection models (like YOLO) say *"The cat is in this box."*
**Segmentation models (U-Net)** say *"These specific pixels belong to the cat."*

U-Net was built for medical imaging but is now the gold standard for any task requiring pixel-perfect accuracy.

## 🏗️ Architecture: The "U" Shape
The architecture looks like a letter **U**:
1.  **Contraction Path (Encoder):** Like VGG/ResNet, it extracts features and shrinks the image size.
2.  **Expansion Path (Decoder):** It upsamples (enlarges) the features back to the original image size.
3.  **Skip Connections:** It copies high-resolution features from the Encoder directly to the Decoder to preserve details (edges/borders).

## 💻 Code Structure (Concept)
```python
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def simple_unet():
    inputs = Input((128, 128, 1))
    
    # 1. Downsample (Encoder)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    # 2. Bottleneck
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    
    # 3. Upsample (Decoder)
    u3 = UpSampling2D((2, 2))(c2)
    # The Magic: Concatenate the earlier layer (c1) with current layer (u3)
    merge3 = concatenate([u3, c1]) 
    c3 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge3)
    
    # 4. Output (Pixel-wise classification)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    return Model(inputs, outputs)
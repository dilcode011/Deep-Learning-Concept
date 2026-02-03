#  YOLO (You Only Look Once): Real-Time Object Detection

**Paper:** [You Only Look Once: Unified, Real-Time Object Detection (2016)](https://arxiv.org/abs/1506.02640)  
**Creators:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

---

##  Introduction
**YOLO** completely changed computer vision by making object detection **fast enough for real-time video**.

Before YOLO, detection systems (like R-CNN) were slow because they treated detection as a classification problem—scanning an image thousands of times to find objects. YOLO treats detection as a **single regression problem**. It looks at the image once (hence the name) and predicts bounding boxes and class probabilities simultaneously.

---

##  How It Works (The "Grid" Concept)
YOLO splits the input image into an **$S \times S$ grid**.

1.  **Responsibility:** If the center of an object falls into a specific grid cell, that cell is "responsible" for detecting that object.
2.  **Prediction:** Each grid cell predicts:
    * **Bounding Boxes ($B$):** The coordinates $(x, y, w, h)$ of the box.
    * **Confidence Score:** How likely is it that the box contains an object? ($Pr(Object) \times IoU$).
    * **Class Probability:** If there is an object, what is it? (Car, Dog, Person, etc.)

---

##  Key Metrics

### 1. Intersection over Union (IoU)
IoU measures how accurate a predicted box is compared to the ground truth (actual) box.
$$IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$
* **IoU = 1:** Perfect match.
* **IoU = 0:** No overlap.

### 2. Non-Max Suppression (NMS)
YOLO often detects the same object multiple times (creating multiple overlapping boxes). NMS cleans this up:
1.  It takes the box with the highest confidence score.
2.  It removes all other boxes that overlap significantly (high IoU) with that "best" box.

---

##  YOLO Versions Evolution

| Version | Key Improvement |
| :--- | :--- |
| **YOLOv1** | The original. Fast (45 fps) but struggled with small objects. |
| **YOLOv2** | Added "Anchor Boxes" and Batch Normalization. Better accuracy. |
| **YOLOv3** | Used "Darknet-53" backbone. Much better at detecting small objects. |
| **YOLOv4/v5** | Optimized for speed and accuracy using PyTorch (Ultralytics). |
| **YOLOv8** | Current state-of-the-art. Anchor-free detection, faster and more accurate. |

---

##  Code Implementation (Using Ultralytics YOLOv8)

Modern YOLO implementation is easiest with the `ultralytics` library.

### 1. Installation
```bash
pip install ultralytics
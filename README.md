[results.csv](https://github.com/user-attachments/files/21958239/results.csv)# Custom YOLO Model Training
## 1. **Dataset Preparation**

* **Photos → Annotations**: First, we collect images and label the objects in them (bounding boxes + class labels). This produces `.txt` annotation files where each line describes:

  ```
  class_id x_center y_center width height
  ```

  (all normalized between 0 and 1).

* These images + annotation files are organized into:

  * **train/** (used to learn)
  * **val/** (used to check if learning is correct)

This separation is important because training only on one set will cause the model to memorize instead of generalizing.

---

## 2. **Training Process**

* **Epoch**: One full pass of all training images through the model. For example, if you have 1000 images and run for 10 epochs, the model sees each image 10 times.

* **Forward Pass**: Images go into the YOLO model → model predicts bounding boxes + class labels.

* **Loss Calculation**: The difference between predicted bounding boxes and true annotations is measured using a *loss function*. It checks:

  * How accurate are the box positions?
  * How correct are the class predictions?

* **Backward Pass (Backpropagation)**: The model adjusts its internal weights (parameters) slightly to reduce the error.

* **Repeat for each batch and each epoch**: Over time, the model improves.

---

## 3. **Validation**

* After each epoch, the model is tested on the **validation set** (data it hasn’t seen while learning).
* This checks whether the model is actually *understanding* or just memorizing.
* Metrics like **mAP (mean Average Precision)**, **precision**, and **recall** are calculated.

---

## 4. **Outputs**

* **Weights File (.pt)**: After training, YOLO saves the learned weights.
* You can use these weights to:

  * Detect objects in new images.
  * Continue training (fine-tuning).

---

## 5. **Simple Analogy (Easy Example)**

Imagine you are teaching a child to recognize **apples vs oranges**:

1. **Dataset**: You show photos of apples and oranges, telling which is which.
2. **Epoch 1**: Child looks at all photos once and guesses → makes many mistakes.
3. **Loss**: You correct them ("No, this is apple, not orange").
4. **Backpropagation**: Child updates memory to reduce mistakes.
5. **Validation**: You test them with new fruit photos they haven’t seen.
6. **More Epochs**: Each time, they get better at distinguishing apples vs oranges.

Finally, the child (model) can identify fruits in completely new photos!

---

✅ That’s how YOLO training works in simple words.

---

# 🔬 Detailed Version (Advanced ML / PhD-Level Explanation)

Now let’s dive deeper into the **machine learning perspective** of YOLO training.

---

## 1. Dataset Representation

* Each image is paired with a set of bounding boxes.
* A bounding box is parameterized as `(x_center, y_center, width, height)` normalized to `[0,1]`.
* Labels are integers representing object categories.
* YOLO expects annotations in this normalized form for scale-invariant training.

**Why normalization?**

* Prevents large images from having disproportionate influence.
* Ensures consistent representation regardless of resolution.

---

## 2. Training Pipeline

### Forward Propagation

* Input: Image → resized (e.g., 640×640).
* CNN Backbone (e.g., CSPDarknet in YOLOv5, or C2f modules in YOLOv8) extracts hierarchical features.
* Neck (PANet/FPN) fuses multi-scale features for small, medium, and large object detection.
* Head outputs predictions: bounding box offsets, objectness score, and class probabilities.

### Loss Function

YOLO uses a **composite loss**:

* **Localization Loss** (bounding box regression):

  * Usually IoU-based (GIoU, DIoU, CIoU).
* **Confidence Loss**: Penalizes incorrect objectness scores.
* **Classification Loss**: Penalizes wrong class predictions.

Mathematically:

```
L_total = λ1 * L_box + λ2 * L_obj + λ3 * L_cls
```

Where:

* `L_box`: IoU-based loss
* `L_obj`: Binary cross-entropy for objectness
* `L_cls`: Cross-entropy loss for classification

### Backpropagation

* Gradients are computed w\.r.t. weights.
* Optimizer (SGD/Adam) updates parameters.
* Learning rate schedules (e.g., cosine decay, one-cycle) are applied for stability.

---

## 3. Validation & Evaluation Metrics

* **Precision (P):** TP / (TP + FP)
* **Recall (R):** TP / (TP + FN)
* **mAP\@0.5:** Mean Average Precision at IoU threshold 0.5.
* **mAP\@0.5:0.95:** Averaged across multiple IoU thresholds (stricter evaluation).

Validation ensures:

* Model is not overfitting (memorizing training set).
* Model generalizes well to unseen data.

---

## 4. Optimization Strategies

* **Transfer Learning**: Start from pretrained COCO weights, fine-tune for custom classes.
* **Data Augmentation**: Mosaic, random crops, flips, HSV augmentation to improve generalization.
* **Batch Normalization**: Stabilizes training by normalizing intermediate outputs.
* **Early Stopping**: Stop when validation performance stops improving.

---

## 5. Deployment Considerations

* **Edge Devices**: Use smaller models (YOLOv8n, YOLOv5s) optimized with quantization or pruning.
* **Cloud/Server**: Larger models (YOLOv8x, YOLOv5x) for high accuracy.
* **Latency vs Accuracy Trade-off**: Must balance performance depending on use case.

---

## 6. Example Command-Line Usage

```bash
# Training
!yolo task=detect mode=train model=yolov8n.pt data=custom.yaml epochs=50 imgsz=640

# Validation
!yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=custom.yaml

# Inference
!yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source="test.jpg"
```

---

## 🎯 Summary

* YOLO is a **single-shot detector** that performs object localization + classification simultaneously.
* Training involves **forward pass, loss calculation, and backpropagation**.
* Validation ensures **generalization** using metrics like mAP.
* Deployment requires trade-offs between **speed and accuracy**.

---
# 🔧 Custom Electronics Detection Model

## Model Overview

I have successfully trained a custom YOLO model for detecting 6 common electronics components used in Arduino and IoT projects. This specialized model can accurately identify and locate the following objects:

### 📋 Detected Classes:
1. **Arduino Uno** - Popular microcontroller board
2. **DHT 11** - Temperature and humidity sensor
3. **ESP 32** - WiFi and Bluetooth microcontroller
4. **HC-SR04** - Ultrasonic distance sensor
5. **LCD 16X2 with I2C** - Character display with I2C interface
6. **Soil Moisture Sensor** - Analog sensor for measuring soil humidity

---

## 📊 Model Performance

### Key Metrics:
- **Overall mAP@0.5**: 0.838 (83.8%)
- **Training Epochs**: 60
- **Best F1 Score**: 0.82 at confidence 0.335

### Individual Class Performance:
| Component | Average Precision (AP) |
|-----------|----------------------|
| Arduino Uno | 0.995 |
| DHT 11 | 0.899 |
| ESP 32 | 0.753 |
| HC-SR04 | 0.682 |
| LCD 16X2 with I2C | 0.828 |
| Soil Moisture Sensor | 0.872 |

---

## 📈 Training Results Analysis

### Loss Curves:
The model showed excellent convergence with:
- **Box Loss**: Decreased from 2.25 to ~0.9
- **Classification Loss**: Reduced from 4.0 to ~0.6
- **DFL Loss**: Minimized from 1.8 to ~1.1

### Performance Insights:
- **Arduino Uno** achieved the highest precision (99.5%) due to its distinctive rectangular shape and USB connector
- **HC-SR04** had the lowest AP (68.2%) as its cylindrical sensors can be confused with similar components
- The model shows robust performance across different lighting conditions and orientations

---

## 🖼️ Sample Detections

The trained model successfully detects electronics components in various scenarios:

### Detection Examples:
- **Multi-component scenes**: Accurately identifies multiple objects simultaneously
- **Different orientations**: Works with components at various angles
- **Varying scales**: Detects both close-up and distant components
- **Mixed backgrounds**: Performs well on different surfaces (fabric, wood, paper)

### Confidence Thresholds:
- **High confidence (>0.8)**: Very reliable detections
- **Medium confidence (0.5-0.8)**: Good detections with minor uncertainty
- **Low confidence (<0.5)**: May require validation

---

## 🎯 Model Applications

This custom YOLO model is perfect for:

### Educational Use:
- **Component identification** in electronics tutorials
- **Inventory management** for maker spaces and labs
- **Quality control** in electronics assembly

### Industrial Applications:
- **Automated sorting** of electronics components
- **Assembly line verification** for Arduino-based projects
- **Stock management** in electronics retail

### Hobbyist Projects:
- **Smart storage systems** that identify components automatically
- **Project documentation** with automatic component labeling
- **Learning aids** for electronics beginners

---

## 💡 Key Features

### Model Strengths:
✅ **High accuracy** on well-defined components like Arduino Uno  
✅ **Robust detection** across different lighting conditions  
✅ **Multi-object detection** in complex scenes  
✅ **Real-time performance** suitable for live applications  

### Areas for Improvement:
🔄 **Small component detection** could be enhanced with more training data  
🔄 **Similar component differentiation** (e.g., different sensor models)  
🔄 **Partially occluded objects** need better handling  

---

## 🚀 Usage Instructions

### Inference Command:
```bash
# Run detection on new images
yolo task=detect mode=predict model=path/to/your/best.pt source="your_image.jpg" conf=0.5

# Batch processing
yolo task=detect mode=predict model=path/to/your/best.pt source="path/to/images/" conf=0.5
```

### Integration Example:
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/your/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for r in results:
    for box in r.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        print(f"Detected: {model.names[class_id]} with {confidence:.2f} confidence")
```

---
# My Model

## Training Files Overview

This model training session generated comprehensive evaluation metrics and visualization files to track performance and analyze results.

## Performance Curves

### BoxF1_curve
The BoxF1_curve represents the F1-score performance metric for bounding box detection throughout the training process. The F1-score is the harmonic mean of precision and recall, providing a balanced measure of model performance that considers both false positives and false negatives.
<img width="2250" height="1500" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/cb3aa85f-20d8-430a-9405-76804ceb1dd7" />


### BoxP_curve
The BoxP_curve tracks precision metrics for bounding box detection, showing how accurately the model identifies positive detections without false positives.
<img width="2250" height="1500" alt="BoxP_curve" src="https://github.com/user-attachments/assets/4b5bafac-8889-4a15-9214-5a3c64d21d6e" />

### BoxPR_curve
The BoxPR_curve displays the Precision-Recall relationship, crucial for understanding the trade-off between precision and recall at different confidence thresholds.
<img width="2250" height="1500" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/406d6a73-7977-43f2-bf9a-e7c528e62dba" />

### BoxR_curve
The BoxR_curve monitors recall performance, indicating how well the model captures all relevant objects in the dataset.
<img width="2250" height="1500" alt="BoxR_curve" src="https://github.com/user-attachments/assets/cb1f8ac3-4647-41d5-9ab5-38905fed2f58" />

## Analysis Files

### Confusion Matrix
- **confusion_matrix**: Raw confusion matrix data showing true vs predicted classifications
<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d2740338-a721-40f1-8597-f6605f3ad72f" />
- **confusion_matrix_normalized**: Normalized version for percentage-based analysis
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/0831d27f-a4a1-4074-b503-f211020f2f22" />


### Labels and Correlogram
- **labels**: Ground truth annotations for training data
![labels](https://github.com/user-attachments/assets/8f27956b-7bc0-4a56-afc9-9caceb7b731e)

- **labels_correlogram**: Visual correlation analysis between different label classes
![labels_correlogram](https://github.com/user-attachments/assets/6a018c80-7d55-4c3a-b310-0c88d5b3935d)

### Training Batches
Multiple training batch visualizations (train_batch0, train_batch1, train_batch2, etc.) showing sample images with annotations during different stages of training.

### Validation Data
- **val_batch0_labels**: Validation set ground truth labels
![val_batch0_labels](https://github.com/user-attachments/assets/82349390-0034-4d86-a998-e5e67fa00da9)

- **val_batch0_pred**: Model predictions on validation set 
![val_batch0_pred](https://github.com/user-attachments/assets/be2fe590-aff6-458a-a199-5ede979358c1)

### Results
- **results** (Excel): Comprehensive metrics and statistics
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/ea877282-9007-41e4-b042-6f02f8d25194" />

- **results** (Image): Visual summary of training results
  [Uploading epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,40.0867,2.24259,4.01549,1.92488,0.2865,0.25594,0.10555,0.05044,1.90119,3.4494,1.97187,8e-05,8e-05,8e-05
2,48.8192,1.79183,2.49735,1.68685,0.6354,0.44869,0.52029,0.24581,1.8536,3.30044,1.90795,0.000167195,0.000167195,0.000167195
3,57.4118,1.77986,1.85557,1.64145,0.62697,0.75775,0.73744,0.34587,1.8843,1.95306,1.85362,0.00025142,0.00025142,0.00025142
4,63.4276,1.73519,1.652,1.61118,0.72481,0.72544,0.74238,0.34492,1.89922,1.5348,2.04077,0.000332675,0.000332675,0.000332675
5,70.4704,1.74262,1.47544,1.59308,0.70868,0.67692,0.70566,0.33972,1.82895,1.49624,1.895,0.00041096,0.00041096,0.00041096
6,76.1846,1.71738,1.38318,1.58883,0.52331,0.66469,0.65713,0.31312,1.91302,1.65223,1.99909,0.000486275,0.000486275,0.000486275
7,82.9348,1.67545,1.2847,1.57713,0.69795,0.71613,0.72407,0.35687,1.9076,1.50055,2.07161,0.00055862,0.00055862,0.00055862
8,88.2223,1.66068,1.25835,1.58926,0.70823,0.70275,0.70471,0.36613,1.88244,1.32971,2.01078,0.000627995,0.000627995,0.000627995
9,95.7984,1.62696,1.1986,1.55095,0.70205,0.6864,0.67217,0.30305,1.90567,1.45345,2.11169,0.0006944,0.0006944,0.0006944
10,100.029,1.65628,1.22414,1.53964,0.73246,0.57037,0.69435,0.3279,1.9473,1.47053,2.19283,0.000757835,0.000757835,0.000757835
11,107.894,1.60353,1.1285,1.50669,0.7873,0.61791,0.71741,0.32355,1.98457,1.34743,2.20663,0.0008183,0.0008183,0.0008183
12,114.315,1.58469,1.08869,1.46656,0.75437,0.48723,0.6356,0.26693,1.85835,1.5738,1.9936,0.0008185,0.0008185,0.0008185
13,120.99,1.60191,1.09928,1.4818,0.83824,0.59032,0.6619,0.27084,1.95258,1.49637,2.07841,0.000802,0.000802,0.000802
14,126.805,1.57431,1.06812,1.47943,0.70958,0.65848,0.66995,0.30089,1.96711,1.37903,2.13059,0.0007855,0.0007855,0.0007855
15,133.786,1.55625,1.03619,1.50057,0.76626,0.71902,0.76469,0.36088,1.83041,1.24764,2.00111,0.000769,0.000769,0.000769
16,140.19,1.51722,1.0029,1.47439,0.79751,0.74286,0.7241,0.34255,1.85142,1.27555,1.99882,0.0007525,0.0007525,0.0007525
17,145.459,1.52998,0.99066,1.46039,0.74135,0.77174,0.7729,0.38451,1.84496,1.1366,1.99303,0.000736,0.000736,0.000736
18,151.84,1.51326,0.94857,1.41408,0.76403,0.74316,0.75642,0.3788,1.88396,1.09424,2.02581,0.0007195,0.0007195,0.0007195
19,157.907,1.53338,0.94771,1.41623,0.74649,0.71244,0.70531,0.35549,1.89334,1.13157,2.02072,0.000703,0.000703,0.000703
20,164.337,1.47386,0.92027,1.40782,0.75994,0.75369,0.75558,0.37192,1.83667,1.12087,1.94341,0.0006865,0.0006865,0.0006865
21,169.442,1.42425,0.94048,1.36939,0.8296,0.73283,0.7819,0.40145,1.81068,1.07702,1.90162,0.00067,0.00067,0.00067
22,175.935,1.37114,0.86094,1.37868,0.79243,0.77762,0.78111,0.41274,1.82197,1.00385,1.93229,0.0006535,0.0006535,0.0006535
23,182.051,1.42284,0.8719,1.41354,0.80175,0.74834,0.78071,0.42816,1.80884,0.98885,1.91552,0.000637,0.000637,0.000637
24,189.294,1.39761,0.83687,1.38646,0.86718,0.79838,0.82077,0.43613,1.77174,0.99206,1.86888,0.0006205,0.0006205,0.0006205
25,195.37,1.38407,0.84196,1.39668,0.82369,0.82032,0.84636,0.41568,1.80152,0.96128,1.88402,0.000604,0.000604,0.000604
26,201.474,1.39169,0.83082,1.36608,0.81468,0.76555,0.79194,0.43664,1.75286,0.99793,1.85694,0.0005875,0.0005875,0.0005875
27,206.377,1.29806,0.79305,1.35649,0.79151,0.78376,0.79877,0.42153,1.76986,0.98697,1.86115,0.000571,0.000571,0.000571
28,213.836,1.37669,0.8303,1.36966,0.74899,0.75379,0.7742,0.37067,1.89264,1.0649,1.98309,0.0005545,0.0005545,0.0005545
29,218.716,1.36107,0.80169,1.34111,0.77612,0.77409,0.77915,0.39096,1.86394,1.04716,1.95322,0.000538,0.000538,0.000538
30,225.192,1.33685,0.79026,1.36032,0.84661,0.74172,0.78883,0.39164,1.84455,1.02798,1.91437,0.0005215,0.0005215,0.0005215
31,231.113,1.30068,0.79773,1.31083,0.7591,0.7818,0.76604,0.40745,1.81117,1.00848,1.88453,0.000505,0.000505,0.000505
32,238.104,1.26886,0.74673,1.31865,0.82939,0.80731,0.83624,0.42417,1.83905,1.07701,1.8841,0.0004885,0.0004885,0.0004885
33,243.854,1.28765,0.7466,1.31512,0.82004,0.85672,0.85389,0.41173,1.84212,1.02425,1.95881,0.000472,0.000472,0.000472
34,250.6,1.23926,0.72521,1.2894,0.87889,0.83606,0.84879,0.43106,1.83952,1.03849,1.91381,0.0004555,0.0004555,0.0004555
35,256.332,1.25561,0.7197,1.2838,0.84466,0.80066,0.82413,0.4108,1.96905,1.00201,2.02264,0.000439,0.000439,0.000439
36,263.086,1.24535,0.74283,1.31701,0.82708,0.81444,0.80919,0.39966,1.92026,0.99878,1.99634,0.0004225,0.0004225,0.0004225
37,268.093,1.26804,0.72852,1.29164,0.81087,0.75487,0.78504,0.40198,1.92069,1.01804,1.99656,0.000406,0.000406,0.000406
38,275.138,1.21997,0.7174,1.26526,0.80614,0.80535,0.8049,0.38554,1.92798,0.98927,2.0417,0.0003895,0.0003895,0.0003895
39,280.675,1.21339,0.71962,1.28892,0.75412,0.79849,0.75689,0.39006,1.92727,0.96195,2.03392,0.000373,0.000373,0.000373
40,286.277,1.23771,0.71911,1.24379,0.78654,0.82662,0.82524,0.4238,1.84501,0.93592,1.93139,0.0003565,0.0003565,0.0003565
41,291.879,1.21342,0.70026,1.2227,0.82401,0.77438,0.80943,0.42189,1.8226,0.93163,1.94421,0.00034,0.00034,0.00034
42,298.119,1.24257,0.69451,1.24058,0.80474,0.80612,0.7984,0.41467,1.86357,0.89239,1.98416,0.0003235,0.0003235,0.0003235
43,304.056,1.18528,0.68234,1.23751,0.80922,0.83395,0.82168,0.43827,1.8343,0.9038,1.95252,0.000307,0.000307,0.000307
44,310.642,1.20643,0.68632,1.25838,0.82905,0.81903,0.80614,0.41587,1.82364,0.91411,1.93407,0.0002905,0.0002905,0.0002905
45,317.253,1.08729,0.64337,1.2131,0.75205,0.84636,0.80638,0.40235,1.86285,0.97408,1.94148,0.000274,0.000274,0.000274
46,322.993,1.1062,0.64178,1.21792,0.80888,0.82358,0.80814,0.38554,1.87957,0.99346,1.95411,0.0002575,0.0002575,0.0002575
47,330.529,1.13119,0.65194,1.18142,0.81209,0.82654,0.80503,0.40128,1.88495,1.0007,1.96189,0.000241,0.000241,0.000241
48,336.253,1.06631,0.61008,1.20149,0.78671,0.83349,0.79394,0.39287,1.87791,1.00083,1.97372,0.0002245,0.0002245,0.0002245
49,342.059,1.06497,0.61676,1.18183,0.73699,0.85293,0.78852,0.40235,1.86929,1.00836,1.97108,0.000208,0.000208,0.000208
50,348.071,1.07281,0.61898,1.18385,0.72519,0.81586,0.76276,0.38835,1.86616,1.01226,1.96741,0.0001915,0.0001915,0.0001915
51,372.164,1.04277,0.59771,1.19532,0.75151,0.77099,0.754,0.39859,1.91557,1.01037,2.00909,0.000175,0.000175,0.000175
52,383.345,1.02782,0.55203,1.16531,0.77196,0.79542,0.78375,0.40085,1.97135,1.00856,2.04791,0.0001585,0.0001585,0.0001585
53,391.776,1.00453,0.53373,1.14956,0.77627,0.80089,0.80133,0.42033,1.92161,0.99354,2.02118,0.000142,0.000142,0.000142
54,398.68,0.98351,0.57472,1.16037,0.7896,0.80825,0.80196,0.42185,1.89238,0.98511,2.00786,0.0001255,0.0001255,0.0001255
55,404.959,0.94753,0.5293,1.14392,0.83316,0.81064,0.8284,0.42609,1.8581,0.94703,1.97588,0.000109,0.000109,0.000109
56,411.211,0.95389,0.52106,1.14717,0.89594,0.78836,0.83756,0.43168,1.84309,0.94775,1.93547,9.25e-05,9.25e-05,9.25e-05
57,416.538,0.91525,0.5002,1.13161,0.88149,0.78868,0.83423,0.42791,1.84377,0.97559,1.93637,7.6e-05,7.6e-05,7.6e-05
58,423.326,0.91708,0.5163,1.12886,0.85232,0.7984,0.83369,0.42754,1.83536,0.96402,1.93549,5.95e-05,5.95e-05,5.95e-05
59,429.075,0.92401,0.50899,1.14019,0.81113,0.84328,0.83823,0.4377,1.81919,0.94351,1.9336,4.3e-05,4.3e-05,4.3e-05
60,434.711,0.8914,0.49403,1.10703,0.80054,0.82802,0.81202,0.42921,1.82289,0.9218,1.94922,2.65e-05,2.65e-05,2.65e-05
results.csv…]()


### Model Weights
**weights** folder containing the trained model parameters and checkpoints.

![File directory view showing machine learning training files including weights folder, performance curve files, confusion matrices, labels, results, and training/validation batch files](image.png)

# Credits & Resources

This project was trained and inspired using the following resource:

- [YOLO Training Notebook by EdjeElectronics](https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=IcoBAeHXa86W)

Special thanks to **EdjeElectronics** for providing an excellent step-by-step guide for training and deploying YOLO models.

---



# Week 7: Neural Networks & Deep Learning Basics

This week focuses on implementing Neural Networks from scratch and using
modern Deep Learning frameworks like TensorFlow and Keras. It covers
fundamental concepts such as forward propagation, backpropagation,
convolutional neural networks, and transfer learning.

---

## 📌 Task 7.1: Multi-Layer Perceptron from Scratch (mlp_scratch.py)

### Objective

Implement a simple neural network using only NumPy.

### Features

- XOR classification problem
- 1 Hidden layer
- Sigmoid activation function
- Forward propagation
- Cross-entropy loss
- Backpropagation (manual gradient computation)
- Gradient descent optimization
- Loss curve visualization
- Decision boundary visualization

### Learning Outcomes

- Understand neural network math
- Apply chain rule in backpropagation
- Implement training loop from scratch

---

## 📌 Task 7.2: Neural Networks with TensorFlow/Keras (keras_neural_network.py)

### Objective

Build a neural network using Keras Sequential API.

### Features

- Breast Cancer dataset classification
- Multiple dense layers
- ReLU & Sigmoid activations
- EarlyStopping callback
- ModelCheckpoint callback
- Training/Validation loss & accuracy plots
- Model saving in:
  - .h5 format
  - SavedModel format
- Model loading and evaluation

### Learning Outcomes

- Use high-level deep learning API
- Prevent overfitting with callbacks
- Save and reload trained models

---

## 📌 Task 7.3: CNN for Image Classification (cnn_image_classification.py)

### Objective

Build a Convolutional Neural Network for Fashion-MNIST classification.

### Features

- Conv2D and MaxPooling layers
- Dropout regularization
- Data augmentation
- Feature map visualization
- Filter visualization
- Confusion matrix evaluation
- TensorFlow Lite (.tflite) conversion

### Learning Outcomes

- Understand CNN architecture
- Apply image augmentation
- Deploy lightweight models for mobile/edge

---

## 📌 Task 7.4: Transfer Learning (transfer_learning.py)

### Objective

Use a pre-trained MobileNetV2 model for custom image classification.

### Features

- Pre-trained MobileNetV2 (ImageNet weights)
- Freeze base layers
- Add custom classification head
- Fine-tuning with low learning rate
- Comparison with model trained from scratch
- Performance comparison table
- Save model in:
  - .h5 format
  - SavedModel format
  - .tflite format

### Learning Outcomes

- Apply transfer learning
- Fine-tune deep neural networks
- Compare scratch vs pre-trained performance

---

## 🛠 Requirements

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn pandas
```

---

## 📊 Summary of Week 7

This week builds a strong foundation in: - Neural network fundamentals -
Deep learning with TensorFlow/Keras - Convolutional Neural Networks -
Transfer Learning - Model deployment formats

---

Author: Abu Zar Farid\
Course: Machine Learning Internship\
W

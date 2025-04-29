# Meteorite Identification Using Deep Learning on NVIDIA Jetson Nano

This project was developed as part of the NASA SEES (STEM Enhancement in Earth Science) Internship Program under the mentorship of Professor Suzanne Foxworth. It uses state-of-the-art computer vision techniques and edge AI hardware to classify aerial and close-range images as either **meteorite** or **non-meteorite**, supporting scientific fieldwork and planetary geology.

## 🌌 Project Overview

The goal of this project is to automate the identification of meteorites using deep learning and deploy the trained model on a portable NVIDIA Jetson Nano device. This enables researchers to use drones or handheld cameras to scan terrain in real-time and receive accurate predictions, dramatically accelerating the meteorite identification process.

We trained a **ResNet-18** convolutional neural network using **transfer learning** from pretrained ImageNet weights, allowing rapid convergence with a limited dataset. After training, the model was exported to **ONNX format** for efficient edge inference.

## 🤖 AI Tools & Techniques

- **PyTorch**: Used for model training and transfer learning.
- **TorchVision**: For data augmentation and model architecture.
- **ONNX**: For exporting the model to an optimized format for inference.
- **ONNX Runtime**: Used for running the exported model on Jetson Nano.
- **NVIDIA Jetson Inference Toolkit**: Accelerated AI framework for edge deployment.
- **Computer Vision**: Image classification to distinguish meteorites from geological lookalikes.
- **Edge AI**: Low-power, on-device inference using Jetson Nano hardware.

---

## 🔧 Hardware Requirements

- NVIDIA Jetson Nano Developer Kit
- MicroSD card (minimum 32GB)
- Ethernet connection (for SSH and Docker setup)
- USB camera (for live image testing, optional)

---
